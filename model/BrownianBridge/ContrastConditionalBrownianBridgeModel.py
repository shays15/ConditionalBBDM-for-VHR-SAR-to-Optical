import torch
import torch.nn as nn
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
import sys
from pathlib import Path

# Add parent directory to path to import mri_contrast_embedder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mri_contrast_embedder import ContrastEmbedding


class ContrastConditionalBrownianBridgeModel(BrownianBridgeModel):
    """
    Modified BBDM for contrast-conditional MRI generation.
    
    Since we're using 'nocond' in the UNet, we condition through the
    Brownian bridge: y is derived from the contrast embedding.
    """
    
    def __init__(self, model_config):
        if hasattr(model_config, 'model'):
            inner_config = model_config.model
        else:
            inner_config = model_config
        
        super().__init__(inner_config)
        
        if hasattr(model_config, 'ContrastEmbedding'):
            contrast_config = model_config.ContrastEmbedding
        else:
            raise AttributeError("Could not find ContrastEmbedding config")
        
        # Create contrast embedding module
        self.contrast_embedder = ContrastEmbedding(
            contrast_dim=contrast_config.contrast_dim,
            embedding_dim=contrast_config.embedding_dim,
            num_contrasts=contrast_config.num_contrasts
        )
        
        self.embedding_dim = contrast_config.embedding_dim
        
        # Create a learnable projection to convert embedding to spatial feature map
        # Use a more sophisticated architecture with layer normalization
        self.embedding_to_latent = nn.Sequential(
            nn.Linear(contrast_config.embedding_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.channels * 64 * 64)  # Project to larger spatial size
        )
        
        # Initialize weights properly
        for module in self.embedding_to_latent:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        print(f"✓ Initialized ContrastConditionalBrownianBridgeModel")
        print(f"  Contrast dim: {contrast_config.contrast_dim}")
        print(f"  Embedding dim: {contrast_config.embedding_dim}")
        print(f"  Using 'nocond' UNet (conditioning via Brownian bridge)")
    
    def get_parameters(self):
        print("✓ Optimizing: Contrast Embedder + Embedding-to-Latent + UNet")
        params = (list(self.denoise_fn.parameters()) + 
                 list(self.contrast_embedder.parameters()) +
                 list(self.embedding_to_latent.parameters()))
        return iter(params)
    
    def forward(self, beta: torch.Tensor, theta: torch.Tensor, context=None):
        """
        Forward pass for training.
        
        Args:
            beta: [B, C, H, W] - source MRI image (used for shape only)
            theta: [B, contrast_dim] - target contrast parameters
            context: Optional pre-computed context (ignored, using theta)
        
        Returns:
            loss: scalar loss value
            log_dict: dictionary with logging information
        """
        B, C, H, W = beta.shape
        device = beta.device
        
        # Get context embedding from contrast parameters
        context_embedding = self.get_contrast_context(theta)  # [B, embedding_dim]
        
        # Project embedding to latent space to create the bridge endpoint
        # This becomes "y" in the Brownian bridge
        y_latent = self.embedding_to_latent(context_embedding)  # [B, C*64*64]
        y_latent = y_latent.reshape(B, self.channels, 64, 64)  # [B, C, 64, 64]
        
        # Resize to match input size if needed
        if (64, 64) != (H, W):
            y_latent = torch.nn.functional.interpolate(
                y_latent, size=(H, W), mode='bilinear', align_corners=False
            )
        
        # Random noise as x0
        x_latent = torch.randn(B, C, H, W, device=device)
        
        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # Call p_losses with:
        # - x0: random latent [B, C, H, W]
        # - y: latent derived from contrast embedding [B, C, H, W]
        # - context: None (since we're using nocond)
        return self.p_losses(x_latent, y_latent, None, t)
    
    def get_contrast_context(self, theta: torch.Tensor) -> torch.Tensor:
        """Convert contrast parameters to context embedding."""
        context = self.contrast_embedder(theta)
        return context
    
    @torch.no_grad()
    def sample(self, theta: torch.Tensor, clip_denoised: bool = True, sample_mid_step: bool = False):
        """Generate MRI images for specified contrasts."""
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        
        B = theta.shape[0]
        device = theta.device
        
        # Get context embedding
        context_embedding = self.get_contrast_context(theta)  # [B, embedding_dim]
        
        # Project to latent space for bridge endpoint
        y_latent = self.embedding_to_latent(context_embedding)  # [B, C*H*W]
        y_latent = y_latent.reshape(B, self.channels, 64, 64)  # [B, C, 64, 64]
        
        # Resize to image size
        y_latent = torch.nn.functional.interpolate(
            y_latent, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False
        )
        
        return self.p_sample_loop(
            y=y_latent,
            context=None,  # No context needed since we use nocond
            clip_denoised=clip_denoised,
            sample_mid_step=sample_mid_step
        )