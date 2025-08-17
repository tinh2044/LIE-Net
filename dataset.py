import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class LowLightDataset(data.Dataset):
    def __init__(self, root, cfg, split="train"):
        self.root = root
        self.cfg = cfg
        self.split = split

        # Setup paths
        if split == "train":
            self.input_dir = os.path.join(root, cfg["train_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["train_dir"], cfg["target_dir"])
        else:
            self.input_dir = os.path.join(root, cfg["test_dir"], cfg["input_dir"])
            self.target_dir = os.path.join(root, cfg["test_dir"], cfg["target_dir"])

        # Get image files
        self.image_files = self._get_image_files()

        # Setup transforms
        self.transform = self._get_transforms()

    def _get_image_files(self):
        """Get list of image files"""
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_files = []

        if os.path.exists(self.input_dir) and os.path.exists(self.target_dir):
            input_files = sorted(
                [
                    f
                    for f in os.listdir(self.input_dir)
                    if f.lower().endswith(valid_extensions)
                ]
            )

            for file in input_files:
                target_file = os.path.join(self.target_dir, file)
                if os.path.exists(target_file):
                    image_files.append(file)

        return image_files

    def _get_transforms(self):
        """Get image transforms"""
        image_size = self.cfg.get("image_size", 256)

        if self.split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        return transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a single sample"""
        filename = self.image_files[idx]

        # Load images
        input_path = os.path.join(self.input_dir, filename)
        target_path = os.path.join(self.target_dir, filename)

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # Apply transforms
        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return {
            "input": input_tensor,
            "target": target_tensor,
            "filename": filename,
            "idx": idx,
        }

    def data_collator(self, batch):
        """Custom collate function for batching"""
        inputs = torch.stack([item["input"] for item in batch])
        targets = torch.stack([item["target"] for item in batch])
        filenames = [item["filename"] for item in batch]
        indices = [item["idx"] for item in batch]

        return {
            "inputs": inputs,
            "targets": targets,
            "filenames": filenames,
            "indices": indices,
        }


def get_training_set(root, cfg):
    """Get training dataset"""
    return LowLightDataset(root, cfg, split="train")


def get_test_set(root, cfg):
    """Get test dataset"""
    return LowLightDataset(root, cfg, split="test")
