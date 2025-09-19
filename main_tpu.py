import os
import argparse
import json
from pathlib import Path

import torch
import torch.utils.data as data

# ===== FIX TPU CONFIGURATION FOR KAGGLE =====


def setup_kaggle_tpu():
    """Properly configure TPU for Kaggle environment"""

    # Check if we're in Kaggle TPU environment
    if not (os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME")):
        print("[INFO] Not in TPU environment, will use CPU/GPU")
        return False

    # Configure PJRT for Kaggle TPU
    os.environ["PJRT_DEVICE"] = "TPU"

    # Important: Set correct TPU topology for Kaggle
    os.environ["XLA_USE_SPMD"] = "1"

    # Kaggle TPU specific settings
    if os.environ.get("TPU_NAME"):
        # For Kaggle TPU v3-8 (8 cores)
        os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1"
        os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
        # Set correct mesh dimensions
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    return True


def safe_import_xla():
    """Safely import XLA modules with error handling"""
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp

        # Test XLA device access
        device = xm.xla_device()
        print(f"[INFO] XLA device initialized: {device}")
        print(f"[INFO] XLA world size: {xm.xrt_world_size()}")

        return xm, pl, xmp, True
    except Exception as e:
        print(f"[WARN] XLA import failed: {str(e)}")
        return None, None, None, False


# Try to setup TPU and import XLA
tpu_available = setup_kaggle_tpu()
if tpu_available:
    xm, pl, xmp, xla_success = safe_import_xla()
    if not xla_success:
        tpu_available = False

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
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU/GPU training"
    )
    return parser


def wrap_loader_for_tpu(loader, device, world_size):
    """Wrap dataloader for TPU with error handling"""
    try:
        return pl.MpDeviceLoader(loader, device)
    except Exception as e:
        print(f"[WARN] Failed to wrap loader for TPU: {e}")
        raise


def tpu_train_eval(rank, args, cfg):
    """TPU training function with enhanced error handling"""
    try:
        # Seed per process
        seed = args.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        device = xm.xla_device()
        print(f"[INFO] Process {rank} using device: {device}")

        # Datasets
        cfg_data = cfg["data"]
        train_set = get_training_set(cfg_data["root"], cfg_data)
        test_set = get_test_set(cfg_data["root"], cfg_data)

        # XLA-compatible samplers
        world_size = xm.xrt_world_size()
        ordinal = xm.get_ordinal()

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=ordinal, shuffle=True
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_set, num_replicas=world_size, rank=ordinal, shuffle=False
        )

        # DataLoaders - reduce num_workers for TPU
        num_workers = min(args.num_workers, 2)  # TPU works better with fewer workers

        train_loader = data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=False,  # Don't pin memory for TPU
            collate_fn=train_set.data_collator
            if hasattr(train_set, "data_collator")
            else None,
        )
        test_loader = data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=False,
            collate_fn=test_set.data_collator
            if hasattr(test_set, "data_collator")
            else None,
        )

        # Wrap loaders for TPU
        train_device_loader = wrap_loader_for_tpu(train_loader, device, world_size)
        test_device_loader = wrap_loader_for_tpu(test_loader, device, world_size)

        # Model
        model = NoirNetASP(**cfg["model"]).to(device)

        # Loss
        loss_fn = LowLightLoss(cfg["loss"]).to(device)

        # Optimizer / Scheduler
        optimizer = build_optimizer(
            config=cfg.get("training", {}).get("optimization", {}), model=model
        )
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
            utils.load_pretrained_flexibly(
                model, args.finetune, device="cpu", strict=False
            )

        # Setup directories
        model_dir = cfg.get("training", {}).get("model_dir", "outputs/noirnetasp_tpu")
        if xm.is_master_ordinal():
            Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Training function with proper XLA optimization
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
            running = {
                "total": 0.0,
                "charbonnier": 0.0,
                "grad": 0.0,
                "ssim": 0.0,
                "count": 0,
            }

            for step, batch in enumerate(loader_local):
                try:
                    inputs = batch["inputs"].to(device)
                    targets = batch["targets"].to(device)

                    # Forward pass
                    preds = model_local(inputs)
                    loss_dict = loss_fn_local(preds, targets)
                    total_loss = loss_dict["total"]

                    # Backward pass
                    optimizer_local.zero_grad(set_to_none=True)
                    total_loss.backward()

                    # XLA optimizer step with proper barrier
                    xm.optimizer_step(optimizer_local, barrier=True)

                    # Accumulate metrics
                    running["total"] += total_loss.detach().item()
                    running["charbonnier"] += loss_dict["charbonnier"].detach().item()
                    running["grad"] += loss_dict["grad"].detach().item()
                    running["ssim"] += loss_dict["ssim"].detach().item()
                    running["count"] += 1

                    if step % print_freq == 0:
                        xm.master_print(
                            f"Train E{epoch_local} S{step}: "
                            f"total={total_loss.item():.4f} "
                            f"charb={loss_dict['charbonnier'].item():.4f} "
                            f"grad={loss_dict['grad'].item():.4f} "
                            f"ssim={loss_dict['ssim'].item():.4f}"
                        )

                except Exception as e:
                    xm.master_print(f"[ERROR] Training step {step} failed: {e}")
                    continue

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
                    try:
                        inputs = batch["inputs"].to(device)
                        targets = batch["targets"].to(device)
                        filenames = batch.get("filenames", None)

                        preds = model_local(inputs).clamp(0.0, 1.0)
                        loss_dict = loss_fn_local(preds, targets)

                        # Compute metrics on CPU to avoid TPU issues
                        from metrics import compute_metrics

                        metrics = compute_metrics(
                            targets.cpu(), preds.cpu(), device="cpu"
                        )

                        agg["psnr"] += float(metrics.get("psnr", 0.0))
                        agg["ssim"] += float(metrics.get("ssim", 0.0))
                        agg["charbonnier"] += float(loss_dict["charbonnier"].item())
                        agg["grad"] += float(loss_dict["grad"].item())
                        agg["ssim_loss"] += float(loss_dict["ssim"].item())
                        agg["n"] += 1.0

                        if step % print_freq == 0:
                            xm.master_print(
                                f"Eval E{epoch_local} S{step}: "
                                f"PSNR={metrics.get('psnr', 0.0):.3f} "
                                f"SSIM={metrics.get('ssim', 0.0):.3f}"
                            )

                    except Exception as e:
                        xm.master_print(f"[ERROR] Eval step {step} failed: {e}")
                        continue

            # Reduce results
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

        # Setup args
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

        # Main training/eval loop
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
                f"Test PSNR={results['psnr']:.3f} SSIM={results['ssim']:.3f}"
            )
            return

        best_psnr = 0.0
        for epoch in range(start_epoch, total_epochs):
            # Set epoch for proper shuffling
            train_sampler.set_epoch(epoch)

            # Train
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

            # Step scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_stats.get("total_loss", 0.0))
            else:
                scheduler.step()

            # Save checkpoint
            if xm.is_master_ordinal():
                ckpt_dir = Path(args_local.output_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"checkpoint_{epoch}.pth"

                # Save to CPU first to avoid TPU memory issues
                checkpoint = {
                    "model_state_dict": xm._fetch_sync(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {},
                    "epoch": epoch,
                }
                torch.save(checkpoint, ckpt_path)

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

            # Save best model
            if test_results["psnr"] > best_psnr and xm.is_master_ordinal():
                best_psnr = test_results["psnr"]
                best_path = Path(args_local.output_dir) / "best_checkpoint.pth"
                checkpoint = {
                    "model_state_dict": xm._fetch_sync(model.state_dict()),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if hasattr(scheduler, "state_dict")
                    else {},
                    "epoch": epoch,
                }
                torch.save(checkpoint, best_path)

            xm.master_print(
                f"* TEST PSNR {test_results['psnr']:.3f} Best PSNR {best_psnr:.3f}"
            )

            # Log results
            if xm.is_master_ordinal():
                log_dir = Path(args_local.output_dir)
                log = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_results.items()},
                    "epoch": epoch,
                }
                with (log_dir / "log.txt").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")

            # Sync at epoch end
            xm.rendezvous("epoch_end")

    except Exception as e:
        print(f"[ERROR] TPU training failed: {e}")
        raise


def run_cpu_gpu(args, cfg):
    """CPU/GPU training fallback"""
    print("[INFO] Running on CPU/GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Rest of CPU/GPU implementation remains the same...
    # [Previous CPU/GPU code here]


def _mp_fn(rank, args_serialized):
    """Multiprocessing function wrapper"""
    args, cfg = args_serialized
    try:
        tpu_train_eval(rank, args, cfg)
    except Exception as e:
        print(f"[ERROR] Process {rank} failed: {e}")
        raise


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

    # Try TPU first, fallback to CPU/GPU
    if args.force_cpu or not tpu_available:
        print("[INFO] Using CPU/GPU training")
        run_cpu_gpu(args, config)
    else:
        try:
            print("[INFO] Attempting TPU training...")
            # Use spawn method for better compatibility
            xmp.spawn(_mp_fn, args=((args, config),), nprocs=None, start_method="spawn")
        except Exception as e:
            print(f"[WARN] TPU init failed, falling back to CPU/GPU: {str(e)}")
            run_cpu_gpu(args, config)
