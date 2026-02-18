import torch
from torch.utils.data import DataLoader
from model.mri_contrast_model import ContrastConditionalBrownianBridgeModel
from mri_multicontrast_dataset import MRIMultiContrastDataset

def train_step(model, batch, optimizer):
    beta = batch['beta'].to(device)          # [B, 1, H, W]
    theta = batch['theta'].to(device)        # [B, contrast_dim]
    target = batch['target'].to(device)      # [B, 1, H, W]
    
    # Forward pass with contrast conditioning
    loss, log_dict = model(beta, theta)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), log_dict


def inference(model, theta_vector, num_samples=1):
    """
    Generate MRI images for specified contrasts.
    
    Args:
        theta_vector: [num_contrasts] - one-hot or continuous contrast parameters
        num_samples: number of samples to generate
    
    Returns:
        images: [num_samples, 1, H, W]
    """
    model.eval()
    with torch.no_grad():
        theta = theta_vector.unsqueeze(0).to(device)  # [1, contrast_dim]
        generated = model.sample(theta, clip_denoised=True)
    return generated