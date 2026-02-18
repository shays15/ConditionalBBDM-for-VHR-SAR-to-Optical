import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import glob

class MRIMultiContrastDataset(Dataset):
    """
    Dataset for multi-contrast MRI synthesis.
    
    Directory structure:
        input/
            subj1_beta_t1.nii.gz
            subj1_beta_t2.nii.gz
            subj2_beta_t1.nii.gz
            subj2_beta_t2.nii.gz
        target/
            subj1_T1.nii.gz
            subj1_T2.nii.gz
            subj2_T1.nii.gz
            subj2_T2.nii.gz
    
    Each sample contains:
    - β: source MRI image (from input directory, always beta)
    - θ: target contrast parameters (T1 or T2, extracted from filename)
    - target: ground truth target contrast image (from target directory)
    """
    
    # Map contrast names to indices
    CONTRAST_TYPES = {
        'T1': torch.tensor([1.0, 0.0]),  # T1 one-hot
        'T2': torch.tensor([0.0, 1.0]),  # T2 one-hot
    }
    
    def __init__(self, input_dir, target_dir):
        """
        Args:
            input_dir: path to input directory (contains beta images)
            target_dir: path to target directory (contains ground truth images)
        """
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        
        # Get all input files (beta images)
        self.input_files = sorted(self.input_dir.glob('*_beta_*.nii.gz'))
        
        if len(self.input_files) == 0:
            raise ValueError(f"No beta MRI files found in {input_dir}. "
                           f"Expected filenames like 'subj1_beta_t1.nii.gz'")
        
        print(f"Found {len(self.input_files)} input (beta) files")
    
    def __len__(self):
        return len(self.input_files)
    
    def _extract_contrast_from_filename(self, filename: str) -> str:
        """
        Extract contrast type (T1 or T2) from filename.
        
        Examples:
            'subj1_beta_t1.nii.gz' -> 'T1'
            'subj1_beta_t2.nii.gz' -> 'T2'
        
        Args:
            filename: the filename to parse
        
        Returns:
            'T1' or 'T2'
        """
        filename_lower = filename.lower()
        
        # Check for T1
        if '_t1_' in filename_lower:
            return 'T1'
        # Check for T2
        elif '_t2_' in filename_lower:
            return 'T2'
        else:
            raise ValueError(f"Filename '{filename}' doesn't contain '_t1_' or '_t2_'")
    
    def _extract_subject_id(self, filename: str) -> str:
        """
        Extract subject ID from filename.
        
        Examples:
            'subj1_beta_t1.nii.gz' -> 'subj1'
            'subj001_beta_t2.nii.gz' -> 'subj001'
        
        Args:
            filename: the input filename
        
        Returns:
            subject ID (everything before '_beta')
        """
        # Remove extension
        base_name = filename.replace('.nii.gz', '').replace('.nii', '')
        
        # Extract subject ID (everything before first '_')
        subject_id = base_name.split('_')[0]       
        slice_id = base_name.split('_')[-1]

        return subject_id, slice_id
    
    def _load_nifti_image(self, file_path: Path) -> torch.Tensor:
        """
        Load NIfTI MRI image from file.
        
        Supports: .nii, .nii.gz
        
        Args:
            file_path: path to the NIfTI file
        
        Returns:
            tensor of shape [1, H, W] (single channel, 2D slice)
        """
        file_path = Path(file_path)
        
        # Load NIfTI file
        img = nib.load(file_path).get_fdata()
        
        # Convert to tensor
        img = torch.from_numpy(np.asarray(img)).float()
        
        # Handle 3D volumes
        if img.ndim == 3:
            # Take middle slice
            mid_slice = img.shape[0] // 2
            img = img[mid_slice]
        elif img.ndim == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Add channel dimension: [H, W] -> [1, H, W]
        img = img.unsqueeze(0)
        
        # Normalize to [-1, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = 2.0 * (img - img_min) / (img_max - img_min) - 1.0
        else:
            # Handle case where all values are the same
            img = torch.zeros_like(img)
        
        return img
    
    def __getitem__(self, idx):
        """
        Get a single MRI sample.
        
        Returns:
            {
                'beta': source MRI from input [1, H, W],
                'theta': contrast vector [2] (one-hot for T1 or T2),
                'target': ground truth from target directory [1, H, W],
                'contrast_name': 'T1' or 'T2',
                'subject_id': subject identifier,
                'input_filename': original input filename,
                'target_filename': corresponding target filename
            }
        """
        input_file_path = self.input_files[idx]
        # print(input_file_path)
        input_filename = input_file_path.name
        
        # Extract subject ID and contrast type
        subject_id, slice_id = self._extract_subject_id(input_filename)
        contrast_name = self._extract_contrast_from_filename(input_filename)
        
        # Load input (beta) image
        beta = self._load_nifti_image(input_file_path)
        
        # Find corresponding target file
        target_pattern = f"{subject_id}*{slice_id}.nii.gz"
        # print(f"Pattern: {target_pattern}")
        # print(f"Target dir: {self.target_dir}")
        target_filenames = glob.glob(str(self.target_dir / target_pattern))        
        if not target_filenames:
            raise FileNotFoundError(
                f"Make sure target directory contains files like: "
                f"'{subject_id}_T1.nii.gz' and '{subject_id}_T2.nii.gz'"
            )
        target_filename = target_filenames[0]
        # print(target_filename)

        # Load target image
        target = self._load_nifti_image(target_filename)
        
        # Create theta as one-hot vector
        theta = self.CONTRAST_TYPES[contrast_name].clone()
        
        return {
            'beta': beta,
            'theta': theta,
            'target': target,
            'contrast_name': contrast_name,
            'subject_id': subject_id,
            'input_filename': input_filename,
            'target_filename': target_filename
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    input_dir = './data/input'
    target_dir = './data/target'
    
    try:
        dataset = MRIMultiContrastDataset(input_dir=input_dir, target_dir=target_dir)
        
        # Get a sample
        sample = dataset[0]
        print("✓ Dataset loaded successfully!")
        print(f"\nSample 0:")
        print(f"  Subject: {sample['subject_id']}")
        print(f"  Contrast: {sample['contrast_name']}")
        print(f"  Beta shape: {sample['beta'].shape}")  # [1, H, W]
        print(f"  Theta: {sample['theta']}")  # [2]
        print(f"  Target shape: {sample['target'].shape}")  # [1, H, W]
        print(f"  Input file: {sample['input_filename']}")
        print(f"  Target file: {sample['target_filename']}")
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        print(f"\n✓ Created dataloader with {len(dataset)} samples")
        
        # Test batch loading
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Beta shape: {batch['beta'].shape}")  # [B, 1, H, W]
            print(f"  Theta shape: {batch['theta'].shape}")  # [B, 2]
            print(f"  Target shape: {batch['target'].shape}")  # [B, 1, H, W]
            print(f"  Subjects: {batch['subject_id']}")
            print(f"  Contrasts: {batch['contrast_name']}")
            if batch_idx >= 1:  # Show first 2 batches
                break
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()