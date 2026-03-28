"""
dataset and dataloader for fundus image classification
(add more augmentation)

Example of the sturcture of the pretrained image dataset (base_dir)
Image naming pattern: {dataset_name}_{image_name}.extension

your/fundus_img/dataset_path/           # Base directory (LAG ACRIMA SIGF train)
├── train/                              # Training set
│   ├── 0/                              # Class 0 (negative)
│   │   ├── dataset1_image001.jpg       # Format: {dataset}_{name}.ext
│   │   ├── dataset2_image002.png
│   │   └── ...
│   └── 1/                              # Class 1 (positive)
│       ├── dataset1_image101.jpg
│       ├── dataset2_image102.png
│       └── ...
├── validation/                         # Validation set (SIGF validation)
│   ├── 0/                              # Class 0 (negative)
│   │   ├── SIGF_val001.jpg
│   │   ├── SIGF_val002.png
│   │   └── ...
│   └── 1/                              # Class 1 (positive)
│       ├── SIGF_val101.jpg
│       ├── SIGF_val102.png
│       └── ...
└── test/                               # Test set  (SIGF test)
    ├── 0/                              # Class 0 (negative)
    │   ├── SIGF_test001.jpg
    │   ├── SIGF_test002.png
    │   └── ...
    └── 1/                              # Class 1 (positive)
        ├── SIGF_test101.jpg
        ├── SIGF_test102.png
        └── ...
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
import re

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as functional
from torchvision import transforms
from PIL import Image
import random


class ClsImgDataSet(Dataset):
    def __init__(self, base_dir=None, split='train', target_size=(256, 256)):
        """
        Initialize the image dataset
        Args:
            base_dir: Root directory where images are located
            split: 'train' or 'test'
            target_size: Target image size (height, width)
            normalize: Whether to perform normalization
        """
        self._base_dir = base_dir
        self.split = split
        self.target_size = target_size
        self.sample_list = []

        # support img type
        self.img_extensions = ('.jpg', '.jpeg', '.png')

        # Get the complete path of the split directory (train or test)
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist!")

        # Iterate through the two label folders 0 and 1
        for label in ['0', '1']:
            label_dir = os.path.join(split_dir, label)
            if not os.path.exists(label_dir):
                continue

            # Get all images in the current label folder
            for file in os.listdir(label_dir):
                if file.lower().endswith(self.img_extensions):
                    # Save the image path and corresponding label
                    self.sample_list.append({
                        'path': os.path.join(label_dir, file),
                        'label': int(label)
                    })

        print(f"Found {len(self.sample_list)} images in {split_dir}")

        # define data augmentation transforms
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.5),  # Random vertical flip
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.3,  # Brightness variation
                    contrast=0.3,  # Contrast variation
                    saturation=0.3,  # Saturation variation
                    hue=0.2  # Hue variation
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(30)  # Random rotation ±30 degrees
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),  # Random translation
                    scale=(0.8, 1.2),  # Random scaling
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=5,  # Gaussian kernel size, must be odd
                    sigma=(0.1, 2.0)  # Standard deviation range for Gaussian kernel
                )
            ], p=0.5),  # 50% probability of applying small Gaussian blur
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=13,  # Gaussian kernel size, must be odd
                    sigma=(0.1, 2.0)  # Standard deviation range for Gaussian kernel
                )
            ], p=0.5),  # 50% probability of applying large Gaussian blur
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        # Test set only needs basic transformations
        self.test_transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sample_list)

    def parse_image_name(self, filename, split='train'):
        """
        Parse image filenames, extract dataset name and image name
        Format: {dataset_name}_{image_name}.extension
        """
        if split == 'train':
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split('/', 1)
            if len(parts) != 2:
                return 'unknown', name_without_ext
            return parts[0], parts[1]
        else:
            return os.path.splitext(filename)[0], os.path.splitext(filename)[0]

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        img_path = sample['path']
        label = sample['label']

        # Parse the filename to get the dataset name and image name
        filename = os.path.basename(img_path)
        src, img_name = self.parse_image_name(filename)

        # read images
        image = Image.open(img_path)

        # augmentation:
        if self.split == 'train':
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)


        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'name': img_name,
            'src': src
        }

# Example of usage：
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = ClsImgDataSet(
        base_dir='your/fundus_img/dataset_path',
        split='train',
        target_size=(256, 256)
    )

    trainloader = DataLoader(dataset, batch_size=1, drop_last=False,shuffle=True, num_workers=4, pin_memory=True)
    for idx, sample in enumerate(trainloader):
        print(idx)
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Image name: {sample['name']}")
        print(f"Src name: {sample['src']}")