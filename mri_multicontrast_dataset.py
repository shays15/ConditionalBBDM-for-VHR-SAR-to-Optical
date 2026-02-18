import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class MRIMultiContrastDataset(Dataset):
    """
    Dataset for multi-contrast MRI synthesis.
    
    Each sample contains:
    - β: source MRI image (any modality)
    - θ: target contrast parameters (one-hot or continuous)
    - target: ground truth target contrast image (if available)
    """
    
    CONTRAST_TYPES = {
        'T1': 0,
        'T2': 1,
        'FLAIR': 2
    }
    
    def __init__(self, data_dir, contrast_list=['T1', 'T2', 'FLAIR']):
        self.data_dir = data_dir
        self.contrast_list = contrast_list
        self.num_contrasts = len(contrast_list)
        # Load your data indices here
        
    def __len__(self):
        # Return total number of samples
        pass
    
    def __getitem__(self, idx):
        """
        Returns:
            {
                'beta': source MRI [1, H, W],
                'theta': contrast vector [num_contrasts],
                'target': ground truth [1, H, W],
                'contrast_name': str
            }
        """
        # Load β (source image) - could be any modality
        beta = np.load(f'{self.data_dir}/images/{idx:04d}_source.npy')
        beta = torch.from_numpy(beta).float()
        
        # Load target contrast type
        contrast_name = self._get_contrast_name(idx)
        
        # Create θ as one-hot vector
        theta = torch.zeros(self.num_contrasts, dtype=torch.float32)
        theta[self.CONTRAST_TYPES[contrast_name]] = 1.0
        
        # Load target image (ground truth)
        target = np.load(f'{self.data_dir}/images/{idx:04d}_{contrast_name}.npy')
        target = torch.from_numpy(target).float()
        
        return {
            'beta': beta,
            'theta': theta,
            'target': target,
            'contrast_name': contrast_name
        }
    
    def _get_contrast_name(self, idx):
        # Your logic to determine which contrast this sample is
        pass