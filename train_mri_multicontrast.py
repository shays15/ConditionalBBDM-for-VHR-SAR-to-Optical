import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import sys
from tqdm import tqdm
import argparse
from argparse import Namespace
from PIL import Image
import numpy as np

from mri_multicontrast_dataset import MRIMultiContrastDataset
from model.BrownianBridge.ContrastConditionalBrownianBridgeModel import ContrastConditionalBrownianBridgeModel


def load_config(config_path):
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def dict_to_namespace(d):
    """Recursively convert dict to argparse.Namespace"""
    if isinstance(d, dict):
        ns = Namespace()
        for k, v in d.items():
            setattr(ns, k, dict_to_namespace(v))
        return ns
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [1, H, W] to PIL Image."""
    tensor = tensor.detach().cpu()
    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np, mode='L')
    return img


def save_validation_batch(batch, predictions, save_dir, epoch):
    """Save validation batch images (beta, target, and predictions)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    beta = batch['beta']
    target = batch['target']
    contrasts = batch['contrast_name']
    subject_ids = batch['subject_id']
    
    B = beta.shape[0]
    
    for b in range(B):
        contrast = contrasts[b] if isinstance(contrasts, (list, tuple)) else contrasts
        subject = subject_ids[b] if isinstance(subject_ids, (list, tuple)) else subject_ids
        
        beta_img = tensor_to_image(beta[b])
        beta_path = save_dir / f'epoch_{epoch:03d}_{subject}_{contrast}_01_beta.png'
        beta_img.save(beta_path)
        
        target_img = tensor_to_image(target[b])
        target_path = save_dir / f'epoch_{epoch:03d}_{subject}_{contrast}_02_target.png'
        target_img.save(target_path)
        
        pred_img = tensor_to_image(predictions[b])
        pred_path = save_dir / f'epoch_{epoch:03d}_{subject}_{contrast}_03_prediction.png'
        pred_img.save(pred_path)


def train_step(model, batch, optimizer, device):
    """Single training step."""
    beta = batch['beta'].to(device)
    theta = batch['theta'].to(device)
    
    loss, log_dict = model(beta, theta)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item(), log_dict


@torch.no_grad()
def validate(model, val_dataloader, device, save_images=False, save_dir=None, epoch=0):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            beta = batch['beta'].to(device)
            theta = batch['theta'].to(device)
            
            loss, _ = model(beta, theta)
            total_loss += loss.item()
            num_batches += 1
            
            if save_images and batch_idx == 0 and save_dir is not None:
                try:
                    context_embedding = model.get_contrast_context(theta)
                    y_latent = model.embedding_to_latent(context_embedding)
                    B, C, H, W = beta.shape
                    predictions = y_latent.reshape(B, C, 64, 64)
                    
                    if (64, 64) != (H, W):
                        predictions = torch.nn.functional.interpolate(
                            predictions, size=(H, W), mode='bilinear', align_corners=False
                        )
                    
                    predictions = torch.clamp(predictions, -1.0, 1.0)
                    save_validation_batch(batch, predictions, save_dir, epoch)
                except Exception as e:
                    print(f"  Warning: Could not save validation images: {e}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(checkpoint, path):
    """Save checkpoint with CPU tensors."""
    checkpoint_cpu = {}
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            checkpoint_cpu[key] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in value.items()
            }
        elif isinstance(value, torch.Tensor):
            checkpoint_cpu[key] = value.cpu()
        else:
            checkpoint_cpu[key] = value
    torch.save(checkpoint_cpu, path)


def train(config_path, input_dir, target_dir, checkpoint_dir=None, result_dir=None, resume_from=None, save_val_images=True):
    """Full training loop."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    if checkpoint_dir is None:
        checkpoint_dir = Path('./checkpoints')
    else:
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    if result_dir is None:
        result_dir = Path('./results')
    else:
        result_dir = Path(result_dir)
    
    if save_val_images:
        (result_dir / 'validation').mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Loading config from: {config_path}")
    config = load_config(config_path)
    config_ns = dict_to_namespace(config)
    
    print(f"\n✓ Loading dataset from:")
    print(f"  Input (beta): {input_dir}")
    print(f"  Target: {target_dir}")
    dataset = MRIMultiContrastDataset(input_dir=input_dir, target_dir=target_dir)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_dataloader)} batches)")
    print(f"  Val: {len(val_dataset)} samples ({len(val_dataloader)} batches)")
    
    print(f"\n✓ Initializing model...")
    model = ContrastConditionalBrownianBridgeModel(config_ns)
    model = model.to(device)
    print(f"✓ Model initialized and moved to {device}")
    
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 1e-6)
    optimizer = optim.Adam(
        model.get_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"✓ Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    
    start_epoch = 0
    if resume_from is not None:
        print(f"\n✓ Loading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✓ Resumed from epoch {start_epoch}")
    
    num_epochs = config['training']['n_epochs']
    best_val_loss = float('inf')
    best_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
        for batch_idx, batch in enumerate(pbar):
            loss, log_dict = train_step(model, batch, optimizer, device)
            train_loss += loss
            avg_loss = train_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        train_loss /= len(train_dataloader)
        
        save_val = save_val_images and ((epoch + 1) % 5 == 0)
        val_loss = validate(
            model, 
            val_dataloader, 
            device, 
            save_images=save_val,
            save_dir=result_dir / 'validation',
            epoch=epoch
        )
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}", end="")
        if save_val:
            print(" ✓ Saved images", end="")
        print()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config
        }
        
        latest_path = checkpoint_dir / 'latest.pt'
        save_checkpoint(checkpoint, latest_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = checkpoint_dir / 'best.pt'
            save_checkpoint(checkpoint, best_path)
            print(f"  ✓ New best model! Val loss: {val_loss:.6f}")
        
        if (epoch + 1) % 50 == 0:
            periodic_path = checkpoint_dir / f'epoch_{epoch+1:04d}.pt'
            save_checkpoint(checkpoint, periodic_path)
    
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch+1} (val_loss: {best_val_loss:.6f})")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MRI Multi-Contrast BBDM')
    parser.add_argument('--config', type=str, default='./configs/MRI-MultiContrast.yaml')
    parser.add_argument('--input_dir', type=str, default='./data/input')
    parser.add_argument('--target_dir', type=str, default='./data/target')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--no_val_images', action='store_true')
    
    args = parser.parse_args()
    train(args.config, args.input_dir, args.target_dir, args.checkpoint_dir, 
          args.result_dir, args.resume_from, not args.no_val_images)