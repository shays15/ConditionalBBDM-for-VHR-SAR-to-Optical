from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import cv2

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # Normalize [-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        
        image = None
        try:
            if str(img_path).endswith((".nii.gz", ".nii")):
                img_np = nib.load(str(img_path)).get_fdata().astype(np.float32)
                #print(f"Img_np: min={img_np.min().item()}, max={img_np.max().item()}")
                #print(f"Shape of loaded NIfTI data: {img_np.shape}")
                if img_np.ndim > 2:
                    img_np = np.squeeze(img_np)
                if img_np.ndim != 2:
                    raise ValueError(f"Expected 2D NIfTI slice, got shape {img_np.shape}")
                if p == 1.0:
                    img_np = np.fliplr(img_np)
                image = Image.fromarray(img_np)
            else:
                image = Image.open(img_path)
                if p == 1.0:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
        except BaseException as e:
            print(f"Failed to load image: {img_path}")
            raise e

        if not image.mode == 'L':
            #image = image.convert('L')
            #print(f"Converting... to 'L'")
            image = Image.fromarray(np.clip(img_np, 0, 1) * 255).convert('L')

        image = transform(image)
        
        #image = None
        #try:
        #    image = Image.open(img_path)
        #except BaseException as e:
        #    print(img_path)

        ##if not image.mode == 'RGB':
        ##    image = image.convert('RGB')
        #if not image.mode == 'L':
        #    image = image.convert('L')


        #image = transform(image)

        if self.to_normal:
            print(f"Before normalization: min={image.min().item()}, max={image.max().item()}")
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            print(f"After normalization: min={image.min().item()}, max={image.max().item()}")

        image_name = Path(img_path).stem
        return image, image_name
