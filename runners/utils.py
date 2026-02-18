import os
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision.utils import make_grid, save_image
from Register import Registers
import sys

import nibabel as nib
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.custom import CustomSingleDataset, CustomAlignedDataset, CustomInpaintingDataset


def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, suffix, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = make_dir(os.path.join(result_path, "samples"))
    sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval"))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Parameter') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'AdamW':
        return torch.optim.AdamW(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


def get_dataset(data_config):
    train_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='train')
    val_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='val')
    test_dataset = Registers.datasets[data_config.dataset_type](data_config.dataset_config, stage='test')
    return train_dataset, val_dataset, test_dataset


# @torch.no_grad()
# def save_single_image(image, save_path, file_name, to_normal=True):
#     image = image.detach().clone()
#     if to_normal:
#         image = image.mul_(0.5).add_(0.5).clamp_(0, 1.)
#     image = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(image)
#     im.save(os.path.join(save_path, file_name))

@torch.no_grad()
def save_single_image(image, save_path, file_name, to_normal=True, min_val=0, max_val=2000, ref_nifti_path='/iacl/pg23/savannah/data/umdctmri/UMDCTMRI-008/00/proc/UMDCTMRI-008_00_00-01_HEAD-CT-UNK-3D-UNK-UNK_n4_reg.nii.gz'):
    """
    Save a PyTorch tensor as a NIfTI file with values scaled to [min_val, max_val],
    using the affine and header from a reference NIfTI.

    Args:
        image (torch.Tensor): Tensor of shape (C, H, W), (H, W), or (H, W, D)
        save_path (str or Path): Directory to save the NIfTI file
        file_name (str): Output filename (e.g., 'pred.nii.gz')
        min_val (float): Minimum intensity value in the saved image
        max_val (float): Maximum intensity value in the saved image
        ref_nifti_path (str or Path): Path to a reference NIfTI file to copy affine and header from
    """
    # Load reference NIfTI to get affine and header
    if ref_nifti_path is not None:
        ref_nii = nib.load(str(ref_nifti_path))
        affine = ref_nii.affine
        header = ref_nii.header
    else:
        affine = np.eye(4)
        header = None

    # Detach from graph and move to CPU
    image = image.detach().cpu().float()

    # Squeeze singleton channel if present
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    elif image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    # Normalize to [0, 1]
    # image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    # Rescale to desired intensity range
    # image = image * (max_val - min_val) + min_val

    # Convert to NumPy
    image_np = image.numpy()
    
    # Rotate 90° clockwise (k=-1), then flip vertically (axis=1)
    if image_np.ndim == 2:
        image_np = np.rot90(image_np, k=1)
        image_np = np.flip(image_np, axis=0)
    elif image_np.ndim == 3:
        # Apply to each 2D slice
        for i in range(image_np.shape[0]):
        	image_np[i] = np.flip(np.rot90(image_np[i], k=1), axis=0)

    # Save as NIfTI
    nii = nib.Nifti1Image(image_np, affine=affine, header=header)
    nib.save(nii, str(Path(save_path) / file_name))

@torch.no_grad()
def get_image_grid(batch, grid_size=4, to_normal=True):
    batch = batch.detach().clone()
    image_grid = make_grid(batch, nrow=grid_size)
    if to_normal:
        image_grid = image_grid.mul_(0.5).add_(0.5).clamp_(0, 1.)
    image_grid = image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return image_grid
