import os
import argparse
import json
from pathlib import Path

import torch
import torch.utils.data as data

# Configure PJRT/XLA before importing torch_xla (Colab/Kaggle TPU)
if "PJRT_DEVICE" not in os.environ:
    if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME"):
        os.environ["PJRT_DEVICE"] = "TPU"
os.environ.setdefault("XLA_USE_SPMD", "1")

# PyTorch/XLA imports
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import yaml
import numpy as np
import random

from net import NoirNetASP
from dataset import get_training_set, get_test_set
from loss import LowLightLoss
from optimizer import build_optimizer, build_scheduler
import utils
from opt import train_one_epoch as train_one_epoch_cpu_gpu
from opt import evaluate_fn as evaluate_fn_cpu_gpu


def get_args_parser():
    parser = argparse.ArgumentParser("Low Light Enhancement (TPU)", add_help=False)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--cfg", type=str, default="configs/lol.yaml")
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    return parser


def wrap_loader_for_tpu(loader, device, world_size):
    return pl.MpDeviceLoader(loader, device)


def tpu_train_eval(rank, args, cfg):
    # Seed per process
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = xm.xla_device()

    # Datasets (built once per process; each sees full dataset; sharding via DistributedSampler)
    cfg_data = cfg["data"]
    train_set = get_training_set(cfg_data["root"], cfg_data)
    test_set = get_test_set(cfg_data["root"], cfg_data)

    # XLA-compatible samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
    )

    # DataLoaders
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=train_set.data_collator
        if hasattr(train_set, "data_collator")
        else None,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=test_set.data_collator
        if hasattr(test_set, "data_collator")
        else None,
    )

    # MpDeviceLoader wraps loader to transparently move tensors to XLA device
    train_device_loader = wrap_loader_for_tpu(train_loader, device, xm.xrt_world_size())
    test_device_loader = wrap_loader_for_tpu(test_loader, device, xm.xrt_world_size())

    # Model
    model = NoirNetASP(**cfg["model"]).to(device)

    # Loss
    loss_fn = LowLightLoss(cfg["loss"]).to(device)

    # Optimizer / Scheduler
    optimizer = build_optimizer(
        config=cfg.get("training", {}).get("optimization", {}), model=model
    )
    # Ensure initial_lr field is set for logging parity
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    total_epochs = args.epochs
    if "training" not in cfg:
        cfg["training"] = {}
    if "optimization" not in cfg["training"]:
        cfg["training"]["optimization"] = {}
    cfg["training"]["optimization"]["total_epochs"] = total_epochs
    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"], optimizer=optimizer, last_epoch=-1
    )

    # Load checkpoints
    start_epoch = 0
    if args.resume:
        ret, missing, unexpected, checkpoint = utils.load_pretrained_flexibly(
            model, args.resume, device="cpu", strict=False
        )
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if (
                hasattr(scheduler, "load_state_dict")
                and "scheduler_state_dict" in checkpoint
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if args.finetune and not args.resume:
        utils.load_pretrained_flexibly(model, args.finetune, device="cpu", strict=False)

    # Logging / dirs
    model_dir = cfg.get("training", {}).get("model_dir", "outputs/noirnetasp_tpu")
    if xm.is_master_ordinal():
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    # XLA: wrap step function for optimizer to ensure device sync
    def train_one_epoch_xla(
        args_local,
        model_local,
        loader_local,
        optimizer_local,
        epoch_local,
        loss_fn_local,
        print_freq=10,
        log_dir="logs",
    ):
        model_local.train()
        # Use plain loop due to custom logger relying on CUDA; we keep simple prints and xm.master_print
        running = {
            "total": 0.0,
            "charbonnier": 0.0,
            "grad": 0.0,
            "ssim": 0.0,
            "count": 0,
        }
        for step, batch in enumerate(loader_local):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            preds = model_local(inputs)
            loss_dict = loss_fn_local(preds, targets)
            total_loss = loss_dict["total"]

            optimizer_local.zero_grad(set_to_none=True)
            total_loss.backward()
            xm.optimizer_step(optimizer_local, barrier=True)

            running["total"] += total_loss.detach().item()
            running["charbonnier"] += loss_dict["charbonnier"].detach().item()
            running["grad"] += loss_dict["grad"].detach().item()
            running["ssim"] += loss_dict["ssim"].detach().item()
            running["count"] += 1

            if step % print_freq == 0:
                xm.master_print(
                    f"Train E{epoch_local} S{step}: total={total_loss.item():.4f} charb={loss_dict['charbonnier'].item():.4f} grad={loss_dict['grad'].item():.4f} ssim={loss_dict['ssim'].item():.4f}"
                )

        # Reduce metrics across cores
        count = torch.tensor(running["count"], device=device, dtype=torch.float32)
        sums = torch.tensor(
            [
                running["total"],
                running["charbonnier"],
                running["grad"],
                running["ssim"],
            ],
            device=device,
            dtype=torch.float32,
        )
        count = xm.all_reduce("sum", count)
        sums = xm.all_reduce("sum", sums)
        reduced = sums / (count + 1e-6)
        stats = {
            "total_loss": reduced[0].item(),
            "charbonnier_loss": reduced[1].item(),
            "grad_loss": reduced[2].item(),
            "ssim_loss": reduced[3].item(),
        }
        return stats

    def evaluate_fn_xla(
        args_local,
        loader_local,
        model_local,
        epoch_local,
        loss_fn_local,
        print_freq=50,
        results_path=None,
        log_dir="logs",
    ):
        model_local.eval()
        agg = {
            "psnr": 0.0,
            "ssim": 0.0,
            "charbonnier": 0.0,
            "grad": 0.0,
            "ssim_loss": 0.0,
            "n": 0.0,
        }
        with torch.no_grad():
            for step, batch in enumerate(loader_local):
                inputs = batch["inputs"].to(device)
                targets = batch["targets"].to(device)
                filenames = batch.get("filenames", None)

                preds = model_local(inputs).clamp(0.0, 1.0)
                loss_dict = loss_fn_local(preds, targets)

                # Compute PSNR/SSIM using existing metrics function if available
                from metrics import compute_metrics

                # Run metrics on CPU to avoid TPU-incompatible ops (e.g., LPIPS)
                metrics = compute_metrics(targets.cpu(), preds.cpu(), device="cpu")
                agg["psnr"] += float(metrics.get("psnr", 0.0))
                agg["ssim"] += float(metrics.get("ssim", 0.0))
                agg["charbonnier"] += float(loss_dict["charbonnier"].item())
                agg["grad"] += float(loss_dict["grad"].item())
                agg["ssim_loss"] += float(loss_dict["ssim"].item())
                agg["n"] += 1.0

                if args_local.save_images and filenames is not None:
                    if xm.is_master_ordinal():
                        from utils import save_eval_images

                        save_eval_images(
                            inputs, preds, targets, filenames, args_local.output_dir
                        )

                if step % print_freq == 0:
                    xm.master_print(
                        f"Eval E{epoch_local} S{step}: PSNR={metrics.get('psnr', 0.0):.3f} SSIM={metrics.get('ssim', 0.0):.3f}"
                    )

        # Reduce
        sums = torch.tensor(
            [
                agg["psnr"],
                agg["ssim"],
                agg["charbonnier"],
                agg["grad"],
                agg["ssim_loss"],
                agg["n"],
            ],
            device=device,
            dtype=torch.float32,
        )
        sums = xm.all_reduce("sum", sums)
        n = sums[5].clamp_min(1.0)
        results = {
            "psnr": (sums[0] / n).item(),
            "ssim": (sums[1] / n).item(),
            "charbonnier_loss": (sums[2] / n).item(),
            "grad_loss": (sums[3] / n).item(),
            "ssim_loss": (sums[4] / n).item(),
        }
        if results_path and xm.is_master_ordinal():
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f)
        return results

    # Attach convenience attrs expected by opt.py-like callers
    class TPUArgs:
        pass

    args_local = TPUArgs()
    args_local.device = device
    args_local.output_dir = cfg.get("training", {}).get(
        "model_dir", "outputs/noirnetasp_tpu"
    )
    args_local.save_images = args.save_images or cfg.get("evaluation", {}).get(
        "save_images", False
    )
    args_local.print_freq = args.print_freq

    # Training/Eval
    if args.eval:
        results = evaluate_fn_xla(
            args_local,
            test_device_loader,
            model,
            epoch=start_epoch,
            loss_fn_local=loss_fn,
            print_freq=args.print_freq,
            results_path=os.path.join(args_local.output_dir, "test_results.json"),
        )
        xm.master_print(
            f"Test loss/metrics on {len(test_loader)} batches: PSNR={results['psnr']:.3f} SSIM={results['ssim']:.3f}"
        )
        return

    best_psnr = 0.0
    for epoch in range(start_epoch, total_epochs):
        # Sampler needs to set epoch for correct shuffling
        train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch_xla(
            args_local,
            model,
            train_device_loader,
            optimizer,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=os.path.join(args_local.output_dir, "log", "train"),
        )

        # Step LR
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_stats.get("total_loss", 0.0))
        else:
            scheduler.step()

        # Save checkpoint (master only)
        if xm.is_master_ordinal():
            ckpt_dir = Path(args_local.output_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_{epoch}.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {},
                    "epoch": epoch,
                },
                ckpt_path,
            )

        # Evaluate
        test_results = evaluate_fn_xla(
            args_local,
            test_device_loader,
            model,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=os.path.join(args_local.output_dir, "log", "test"),
        )

        # Save best
        if test_results["psnr"] > best_psnr and xm.is_master_ordinal():
            best_psnr = test_results["psnr"]
            best_path = Path(args_local.output_dir) / "best_checkpoint.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {},
                    "epoch": epoch,
                },
                best_path,
            )
        xm.master_print(
            f"* TEST PSNR {test_results['psnr']:.3f} Best PSNR {best_psnr:.3f}"
        )

        # Simple log.txt (master only)
        if xm.is_master_ordinal():
            log_dir = Path(args_local.output_dir)
            log = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_results.items()},
                "epoch": epoch,
            }
            with (log_dir / "log.txt").open("a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

        # Ensure all cores sync at epoch end
        xm.rendezvous("epoch_end")


def run_cpu_gpu(args, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Seeds
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cfg_data = cfg["data"]
    train_set = get_training_set(cfg_data["root"], cfg_data)
    test_set = get_test_set(cfg_data["root"], cfg_data)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=train_set.data_collator
        if hasattr(train_set, "data_collator")
        else None,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=test_set.data_collator
        if hasattr(test_set, "data_collator")
        else None,
    )

    model = NoirNetASP(**cfg["model"]).to(device)
    loss_fn = LowLightLoss(cfg["loss"]).to(device)
    optimizer = build_optimizer(
        config=cfg.get("training", {}).get("optimization", {}), model=model
    )
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]
    if "training" not in cfg:
        cfg["training"] = {}
    if "optimization" not in cfg["training"]:
        cfg["training"]["optimization"] = {}
    cfg["training"]["optimization"]["total_epochs"] = args.epochs
    scheduler, _ = build_scheduler(
        config=cfg["training"]["optimization"], optimizer=optimizer, last_epoch=-1
    )

    start_epoch = 0
    if args.resume:
        ret, missing, unexpected, checkpoint = utils.load_pretrained_flexibly(
            model, args.resume, device="cpu", strict=False
        )
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if (
                hasattr(scheduler, "load_state_dict")
                and "scheduler_state_dict" in checkpoint
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if args.finetune and not args.resume:
        utils.load_pretrained_flexibly(model, args.finetune, device="cpu", strict=False)

    out_dir = Path(
        cfg.get("training", {}).get("model_dir", "outputs/noirnetasp_cpu_gpu")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    class LocalArgs:
        pass

    local_args = LocalArgs()
    local_args.device = device
    local_args.output_dir = str(out_dir)
    local_args.save_images = args.save_images or cfg.get("evaluation", {}).get(
        "save_images", False
    )
    local_args.print_freq = args.print_freq

    if args.eval:
        results = evaluate_fn_cpu_gpu(
            local_args,
            test_loader,
            model,
            epoch=start_epoch,
            loss_fn=loss_fn,
            print_freq=args.print_freq,
            results_path=str(out_dir / "test_results.json"),
            log_dir=str(out_dir / "log_eval_test"),
        )
        print(
            f"Eval: PSNR={results.get('psnr', 0.0):.3f} SSIM={results.get('ssim', 0.0):.3f}"
        )
        return

    best_psnr = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch_cpu_gpu(
            local_args,
            model,
            train_loader,
            optimizer,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=str(out_dir / "log" / "train"),
        )
        scheduler.step()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
                if hasattr(scheduler, "state_dict")
                else {},
                "epoch": epoch,
            },
            out_dir / f"checkpoint_{epoch}.pth",
        )

        test_results = evaluate_fn_cpu_gpu(
            local_args,
            test_loader,
            model,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=str(out_dir / "log" / "test"),
        )

        if test_results.get("psnr", 0.0) > best_psnr:
            best_psnr = test_results["psnr"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {},
                    "epoch": epoch,
                },
                out_dir / "best_checkpoint.pth",
            )
        print(
            f"* TEST PSNR {test_results.get('psnr', 0.0):.3f} Best PSNR {best_psnr:.3f}"
        )


def _mp_fn(rank, args_serialized):
    args, cfg = args_serialized
    tpu_train_eval(rank, args, cfg)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(
        "Low Light Enhancement TPU", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    with open(args.cfg, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    out_dir = Path(
        config.get("training", {}).get("model_dir", "outputs/noirnetasp_tpu")
    )
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Try TPU via PJRT first; fall back to CPU/GPU if TPU init fails.
    try:
        # 'spawn' is safer than 'fork' across environments
        xmp.spawn(_mp_fn, args=((args, config),), nprocs=None, start_method="spawn")
    except Exception as e:
        print(
            "[WARN] TPU init failed, falling back to CPU/GPU single-process training."
        )
        print(str(e))
        run_cpu_gpu(args, config)
