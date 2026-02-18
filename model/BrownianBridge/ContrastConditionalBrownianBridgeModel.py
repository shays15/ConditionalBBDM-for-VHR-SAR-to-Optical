import torch
import torch.nn as nn
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from mri_contrast_embedder import ContrastEmbedding

class ContrastConditionalBrownianBridgeModel(BrownianBridgeModel):
    """
    Modified for contrast-conditional generation.
    
    Instead of conditioning on another image (β), we condition on
    a contrast parameter vector (θ) that specifies what type of
    MRI contrast to synthesize.
    """
    
    # def __init__(self, model_config):
    #     super().__init__(model_config)
    def __init__(self, model_config):
        # Don't call super().__init__() for the full BBDM init
        # Instead, initialize components selectively
        nn.Module.__init__(self)
        # ... initialize components separately

        # Remove VQGAN since we're not conditioning on an image
        # self.vqgan is only used for encoding β (source image)
        
        # Create contrast embedding module
        self.contrast_embedder = ContrastEmbedding(
            contrast_dim=model_config.ContrastEmbedding.contrast_dim,
            embedding_dim=model_config.ContrastEmbedding.embedding_dim,
            num_contrasts=model_config.ContrastEmbedding.num_contrasts
        )
        
        self.condition_key = model_config.BB.params.UNetParams.condition_key
        
    def get_parameters(self):
        """
        Optimize both the UNet and the contrast embedder.
        """
        print("get parameters to optimize: Contrast Embedder, UNet")
        params = list(self.denoise_fn.parameters()) + list(self.contrast_embedder.parameters())
        return iter(params)
    
    
    def forward(self, beta: torch.Tensor, theta: torch.Tensor, context=None):
        """
        Args:
            beta: [B, C, H, W] - source MRI image
            theta: [B, contrast_dim] - target contrast parameters
            context: Optional pre-computed context
        
        Returns:
            loss and log_dict
        """
        # β is the source image we want to translate FROM
        # θ specifies what contrast to synthesize TO
        
        # In the BBDM formulation:
        # x0 = target (what we want to synthesize, initially random)
        # y = condition (what guides the generation)
        
        # For your case, we generate all outputs from θ conditioning
        # We can optionally also use β for additional guidance
        
        with torch.no_grad():
            # We don't need to encode β as a condition anymore
            # But we might want to encode it for feature extraction
            # beta_latent = self.encode_source(beta, cond=False)  # Optional
            pass
        
        # Get context from contrast parameters
        context = self.get_contrast_context(theta)
        
        # Random target latent (will be denoised towards theta)
        B, C, H, W = beta.shape
        x_latent = torch.randn_like(beta)  # Random initialization
        
        # Get timesteps
        device = beta.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()
        
        # For BBDM: we bridge from x (random) to y (condition-based)
        # y_latent should represent the "endpoint" encoded from θ
        # One approach: use a learned "anchor" or the θ embedding itself
        
        # return self.p_losses(x_latent, theta, context, t)
        # Create a learned or projected "endpoint" from theta
        # y_latent = self.contrast_embedder.spatial_proj(context)
        y_latent = y_latent.reshape(B, 1, 16, 16)  # Or appropriate spatial dims
        return self.p_losses(x_latent, y_latent, context, t)
    
    def get_contrast_context(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Convert contrast parameters to context embedding.
        
        Args:
            theta: [B, contrast_dim]
        
        Returns:
            context: [B, embedding_dim] - learnable embedding
        """
        context = self.contrast_embedder(theta)
        return context
    
    @torch.no_grad()
    def sample(self, theta: torch.Tensor, num_samples: int = 1, 
               clip_denoised=False, sample_mid_step=False):
        """
        Generate MRI images for specified contrasts.
        
        Args:
            theta: [B, contrast_dim] or [contrast_dim] - contrast parameters
            num_samples: number of samples to generate per contrast
        
        Returns:
            generated images
        """
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)  # [contrast_dim] -> [1, contrast_dim]
        
        B = theta.shape[0]
        context = self.get_contrast_context(theta)
        
        # Initialize y from context (the "bridge endpoint")
        # Option 1: Use context directly
        y = context
        
        # Option 2: Project context to spatial dimensions
        # y = self.contrast_embedder.to_spatial(context, self.image_size)
        
        return self.p_sample_loop(y=y, context=context, 
                                 clip_denoised=clip_denoised, 
                                 sample_mid_step=sample_mid_step)