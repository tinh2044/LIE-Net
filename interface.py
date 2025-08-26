import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from net import LIENet
from utils import save_img, get_model_info


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Low Light Image Enhancement Inference"
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/low_light.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--pretrained_model", type=str, required=True, help="Pretrained model path"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing low light images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inference/",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="inference batch size"
    )
    parser.add_argument(
        "--show_flops_params",
        action="store_true",
        help="Show number of flops and parameter of model",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    return parser


def main(args):
    # Load config
    with open(args.cfg_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    model = LIENet(**config["model"]).to(device)

    # Load pretrained weights
    print(f"Loading model from {args.pretrained_model}")
    checkpoint = torch.load(args.pretrained_model, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print("Model loaded successfully")

    # Show model info if requested
    if args.show_flops_params:
        input_shape = (1, 3, config["data"]["image_size"], config["data"]["image_size"])
        model_info = get_model_info(model, input_shape, device)

        print("Model Information:")
        print(f"  Total parameters: {model_info['total_params']:,}")
        print(f"  Trainable parameters: {model_info['trainable_params']:,}")

        if "flops" in model_info:
            print(f"  FLOPs: {model_info['flops_str']}")
            print(f"  MACs: {model_info['macs_str']}")
            print(f"  Parameters (from thop): {model_info['params_str']}")
        print()

    # Create dataset for inference
    # Modify config for inference
    inference_config = config["data"].copy()
    inference_config["root"] = args.input_dir
    inference_config["test_dir"] = ""
    inference_config["input_dir"] = ""
    inference_config["target_dir"] = ""

    # Create a simple dataset for inference
    from PIL import Image
    import torchvision.transforms as transforms

    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, input_dir, image_size=256):
            self.input_dir = input_dir
            self.image_size = image_size
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

            # Get all image files
            valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
            self.image_files = [
                f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)
            ]
            self.image_files.sort()

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            filename = self.image_files[idx]
            image_path = os.path.join(self.input_dir, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)
            return image_tensor, filename

    # Create dataset and dataloader
    inference_dataset = InferenceDataset(args.input_dir, config["data"]["image_size"])
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Found {len(inference_dataset)} images for inference")

    # Inference loop
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, filenames) in enumerate(inference_loader):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                pred_l, pred_h = outputs
            else:
                pred_l = outputs
                pred_h = outputs

            # Save results
            for i, filename in enumerate(filenames):
                # Save enhanced image
                output_path = os.path.join(args.output_dir, f"enhanced_{filename}")
                save_img(pred_l[i].cpu(), output_path)

                # Save high resolution if available
                if isinstance(outputs, tuple):
                    hr_output_path = os.path.join(
                        args.output_dir, f"enhanced_hr_{filename}"
                    )
                    save_img(pred_h[i].cpu(), hr_output_path)

                print(f"Processed: {filename}")

    print(f"Inference completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Low Light Enhancement Inference", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
