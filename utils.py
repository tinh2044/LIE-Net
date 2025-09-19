import os
import torch
import torch.distributed as dist
import numpy as np
from PIL import Image


def save_img(image_tensor, filename):
    """Save image tensor to file"""
    image_numpy = image_tensor.detach().float().cpu().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)


def count_model_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops(model, input_shape, device):
    from fvcore.nn import FlopCountAnalysis

    input_tensor = torch.randn(input_shape).to(device)
    flops = FlopCountAnalysis(model, input_tensor).total()
    return flops


def get_model_info(model, input_shape, device):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }

    flops = calculate_flops(model, input_shape, device)
    if flops is not None:
        info["flops"] = flops
        info["flops_str"] = f"{flops / 1e9:.2f}G"
    return info


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get world size for distributed training"""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get rank for distributed training"""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if current process is main process"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Save only on main process"""
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_distributed():
    """Initialize distributed training"""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        print(
            f"Setting up distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set the device for this process
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def check_state_dict(model, state_dict):
    """Check if model and state dict are compatible"""
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())

    # Check if all model keys are in state dict
    missing_keys = model_keys - state_dict_keys
    if missing_keys:
        print(f"Missing keys in state dict: {missing_keys}")

    match_key = model_keys & state_dict_keys
    if match_key:
        print(f"Matching keys in state dict: {match_key}")
    unexpected_keys = state_dict_keys - model_keys
    if unexpected_keys:
        print(f"Unexpected keys in state dict: {unexpected_keys}")

    if missing_keys or unexpected_keys:
        return False

    return True


def save_checkpoints(model, optimizer, scheduler, epoch, loss, path_dir, name=None):
    """Save training checkpoints"""
    os.makedirs(path_dir, exist_ok=True)

    if name is None:
        name = f"checkpoint_epoch_{epoch}.pth"

    checkpoint_path = os.path.join(path_dir, name)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def load_checkpoints(model, optimizer, scheduler, path, resume=True):
    """Load training checkpoints"""
    if not os.path.exists(path):
        print(f"Checkpoint not found: {path}")
        return 0, 0.0

    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if resume and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if resume and scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")

    return epoch, loss


def save_sample_images(inputs, pred, targets, batch_idx, epoch, output_dir):
    # Save first image in batch
    input_img = inputs[0]
    pred_img = pred[0]
    target_img = targets[0]

    # Concatenate images horizontally
    combined = torch.cat([input_img, pred_img, target_img], dim=2)

    filename = os.path.join(output_dir, f"sample_{batch_idx}_{epoch}.png")
    save_img(combined, filename)

    # Create labeled version with text
    # save_labeled_sample_images(inputs, pred, targets, batch_idx, epoch, output_dir)


def save_eval_images(inputs, pred, targets, filenames, output_dir):
    """Save evaluation images"""
    save_dir = os.path.join(
        output_dir,
        "eval",
    )
    os.makedirs(save_dir, exist_ok=True)

    for i, filename in enumerate(filenames):
        input_img = inputs[i]
        pred_img = pred[i]
        target_img = targets[i]

        combined = torch.cat([input_img, pred_img, target_img], dim=2)
        combined_path = os.path.join(save_dir, f"{filename}_combined.png")
        save_img(combined, combined_path)


def _maybe_strip_prefix(state_dict, prefix: str):
    """Return a copy of state_dict with a leading prefix stripped from all keys
    when every key starts with that prefix. Otherwise returns the original dict.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(k.startswith(prefix) for k in keys):
        plen = len(prefix)
        return {k[plen:]: v for k, v in state_dict.items()}
    return state_dict


def extract_model_state_dict(checkpoint_or_state):
    """Extract the model's state_dict from a checkpoint dict or return the dict itself.

    Args:
        checkpoint_or_state (dict): A checkpoint dict or a model state dict.

    Returns:
        dict: The model state dict.
    """
    obj = checkpoint_or_state
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def align_and_filter_state_dict(model, state_dict):
    """Align keys to the target model and filter out non-matching entries.
    """
    raw_sd = extract_model_state_dict(state_dict)
    # Normalize common prefixes
    sd = _maybe_strip_prefix(raw_sd, "module.")
    sd = _maybe_strip_prefix(sd, "model.")

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())

    # Keep only overlapping keys
    filtered = {k: v for k, v in sd.items() if k in model_keys}

    missing = model_keys - set(filtered.keys())
    unexpected = sd_keys - model_keys
    return filtered, missing, unexpected


def load_pretrained_flexibly(model, checkpoint_or_path, device="cpu", strict=False):
    """Load weights into model handling DDP prefixes and key mismatches gracefully.
    """
    if isinstance(checkpoint_or_path, str):
        try:
            checkpoint = torch.load(
                checkpoint_or_path, map_location=device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_or_path, map_location=device)
    else:
        checkpoint = checkpoint_or_path

    state_dict = extract_model_state_dict(checkpoint)
    aligned, missing, unexpected = align_and_filter_state_dict(model, state_dict)
    ret = model.load_state_dict(aligned, strict=strict)
    return ret, missing, unexpected, checkpoint
