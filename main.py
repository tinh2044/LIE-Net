import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import time
import argparse
import json
import datetime
import numpy as np
import yaml
import random
from pathlib import Path
from loguru import logger


from loss import LowLightLoss
from optimizer import build_optimizer, build_scheduler
from dataset import get_training_set, get_test_set
from net import LIENet
from opt import train_one_epoch, evaluate_fn
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Low Light Image Enhancement Training", add_help=False
    )
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=800, type=int)

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/lol.yaml",
        help="Path to config file",
    )

    parser.add_argument("--print_freq", default=1, type=int, help="print frequency")

    return parser


def main(args, cfg):
    model_dir = cfg["training"]["model_dir"]
    log_dir = f"{model_dir}/log"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    utils.init_distributed_mode(args)

    seed = args.seed + utils.get_rank()
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    # Create datasets
    cfg_data = cfg["data"]
    train_data = get_training_set(cfg_data["root"], cfg_data)
    test_data = get_test_set(cfg_data["root"], cfg_data)

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.data_collator,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.data_collator,
        pin_memory=True,
    )

    # Create model
    model = LIENet(**cfg["model"])
    model = model.to(device)
    n_parameters = utils.count_model_parameters(model)

    # Create loss function
    loss_weights = cfg["model"]["loss"]
    loss_fn = LowLightLoss(loss_weights, device=device)
    print(f"Number of parameters: {n_parameters}")

    # Calculate model info
    input_shape = (args.batch_size, 3, cfg_data["image_size"], cfg_data["image_size"])
    model_info = utils.get_model_info(model, input_shape, device)

    print("Model Information:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")

    if "flops" in model_info:
        print(f"  FLOPs: {model_info['flops_str']}")
        print(f"  MACs: {model_info['macs_str']}")
        print(f"  Parameters (from thop): {model_info['params_str']}")
    print()

    # Load pretrained model if specified
    if args.finetune:
        print(f"Finetuning from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        ret = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    # Create optimizer and scheduler
    optimizer = build_optimizer(config=cfg["training"]["optimization"], model=model)

    # Set initial_lr for optimizer (needed for scheduler resume)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    # Update config with total epochs for warmup scheduler
    cfg["training"]["optimization"]["total_epochs"] = args.epochs

    # Initialize scheduler with correct last_epoch for resume
    scheduler_last_epoch = -1
    if args.resume:
        print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if utils.check_state_dict(model, checkpoint["model_state_dict"]):
            ret = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            print("Model and state dict are different")
            raise ValueError("Model and state dict are different")

        if "epoch" in checkpoint:
            scheduler_last_epoch = checkpoint["epoch"]
        args.start_epoch = checkpoint["epoch"] + 1
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    # Create scheduler with correct last_epoch
    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"],
        optimizer=optimizer,
        last_epoch=scheduler_last_epoch,
    )

    # Load optimizer and scheduler state if resuming
    if args.resume:
        if (
            not args.eval
            and "optimizer_state_dict" in checkpoint
            and "scheduler_state_dict" in checkpoint
        ):
            print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"New learning rate : {scheduler.get_last_lr()[0]}")

    # Add loss function and output directory to args
    args.output_dir = model_dir
    args.save_images = cfg["evaluation"]["save_images"]

    output_dir = Path(cfg["training"]["model_dir"])

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
            )

        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch=0,
            loss_fn=loss_fn,
            print_freq=args.print_freq,
            results_path=f"{model_dir}/test_results.json",
            log_dir=f"{log_dir}/eval/test",
        )
        print(
            f"Test loss of the network on the {len(test_dataloader)} test images: {test_results['psnr']:.3f} PSNR"
        )
        print(f"* TEST SSIM {test_results['ssim']:.3f}")
        return

    print(f"Training on {device}")
    print(
        f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
    )
    start_time = time.time()
    best_psnr = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_results = train_one_epoch(
            args,
            model,
            train_dataloader,
            optimizer,
            scheduler,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=f"{log_dir}/train",
        )
        scheduler.step()

        # Save checkpoint
        checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
        prev_chkpt = output_dir / f"checkpoint_{epoch - 1}.pth"
        if os.path.exists(prev_chkpt):
            os.remove(prev_chkpt)
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        print()

        # Evaluate
        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch,
            loss_fn,
            print_freq=args.print_freq,
            log_dir=f"{log_dir}/test",
        )

        # Save best model
        if test_results["psnr"] > best_psnr:
            best_psnr = test_results["psnr"]
            checkpoint_paths = [output_dir / "best_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

        print(f"* TEST PSNR {test_results['psnr']:.3f} Best PSNR {best_psnr:.3f}")

        # Log results
        log_results = {
            **{f"train_{k}": v for k, v in train_results.items()},
            **{f"test_{k}": v for k, v in test_results.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }
        print()
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_results) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "Low Light Enhancement", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    with open(args.cfg, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
