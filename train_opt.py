import torch
import torch.nn.functional as F
import os
from utils import save_img


def train_one_epoch(model, data_loader, optimizer, losses, device, epoch, args):
    """Train for one epoch with detailed logging"""
    model.train()
    total_loss = 0.0
    total_loss_L = 0.0
    total_loss_H = 0.0

    for iteration, batch in enumerate(data_loader, 1):
        rgb, tarL, tarH, indx = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
            batch[3],
        )

        fake_b, fake_b1 = model(rgb)
        optimizer.zero_grad()

        loss_g_L = losses["weighted_loss4"](
            losses["charbonnier_loss"](tarL, fake_b),
            losses["perceptual_loss"](fake_b, tarL),
            losses["gradient_loss"](fake_b, tarL),
            losses["ms_ssim_loss"](fake_b, tarL),
        )

        loss_g_H = losses["weighted_loss4"](
            losses["charbonnier_loss"](tarH, fake_b1),
            losses["perceptual_loss"](fake_b1, tarH),
            losses["gradient_loss"](fake_b1, tarH),
            losses["ms_ssim_loss"](fake_b1, tarH),
        )

        loss_g = losses["weighted_loss2"](loss_g_L, loss_g_H)

        loss_g.backward()
        optimizer.step()

        total_loss += loss_g.item()
        total_loss_L += loss_g_L.item()
        total_loss_H += loss_g_H.item()

        if iteration % args.print_freq == 0:
            output_dir_train = os.path.join(args.output_dir, "images_train")
            os.makedirs(output_dir_train, exist_ok=True)
            out_image = torch.cat((rgb, fake_b, tarL), 3)
            save_img(
                out_image[0].detach().cpu(),
                f"{output_dir_train}/epoch_{epoch}_iter_{iteration}.png",
            )

            print(
                f"Epoch [{epoch:3d}]({iteration:4d}/{len(data_loader):4d}): "
                f"Loss_Total: {loss_g.item():.6f}, "
                f"Loss_L: {loss_g_L.item():.6f}, "
                f"Loss_H: {loss_g_H.item():.6f}"
            )

    avg_loss = total_loss / len(data_loader)
    avg_loss_L = total_loss_L / len(data_loader)
    avg_loss_H = total_loss_H / len(data_loader)

    return {"loss": avg_loss, "loss_L": avg_loss_L, "loss_H": avg_loss_H}


def evaluate_model(model, data_loader, device, epoch, args, save_images=True):
    """Evaluate model and optionally save result images"""
    model.eval()

    if save_images:
        output_dir = os.path.join(args.output_dir, "images_test")
        os.makedirs(output_dir, exist_ok=True)

    total_samples = 0

    with torch.no_grad():
        for test_iter, batch in enumerate(data_loader, 1):
            rgb_input, target, targetH, idx = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
                batch[3],
            )

            prediction_L, prediction_H = model(rgb_input)

            if save_images:
                out = torch.cat((rgb_input, prediction_L, target), 3)
                filename = f"{output_dir}/epoch_{epoch}_sample_{idx[0]}.png"
                save_img(out[0].detach().cpu(), filename)

                if args.save_high_res:
                    hr_filename = f"{output_dir}/epoch_{epoch}_sample_{idx[0]}_HR.png"
                    save_img(prediction_H[0].detach().cpu(), hr_filename)

            total_samples += rgb_input.size(0)

    print(f"Evaluation completed on {total_samples} samples")
    return {"samples_processed": total_samples}


def calculate_metrics(pred, target):
    """Calculate image quality metrics like PSNR, SSIM"""
    return {"psnr": 0.0, "ssim": 0.0}


def save_training_state(model, optimizer, scheduler, epoch, loss, args, is_best=False):
    """Save training state with better organization"""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pth")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "args": args,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    if not is_best and epoch > 3:
        old_checkpoint = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch - 3}.pth"
        )
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)

    return checkpoint_path


def load_training_state(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training state from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")

    return epoch, loss


def print_training_summary(epoch, train_stats, eval_stats, lr, elapsed_time):
    """Print formatted training summary"""
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch:3d} SUMMARY")
    print("=" * 80)
    print(f"Training Loss:   {train_stats['loss']:.6f}")
    print(f"  - Loss L:      {train_stats.get('loss_L', 0):.6f}")
    print(f"  - Loss H:      {train_stats.get('loss_H', 0):.6f}")
    print(f"Learning Rate:   {lr:.8f}")
    print(f"Epoch Time:      {elapsed_time:.2f}s")
    if eval_stats:
        print(
            f"Evaluation:      {eval_stats.get('samples_processed', 0)} samples processed"
        )
    print("=" * 80 + "\n")
