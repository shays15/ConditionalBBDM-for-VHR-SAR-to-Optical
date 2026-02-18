import torch
import torch.nn as nn
import numpy as np

class ContrastEmbedding(nn.Module):
    """
    Converts contrast parameter vectors θ into learnable embeddings.
    
    θ can be:
    - One-hot encoded: [1,0,0,0] for T1, [0,1,0,0] for T2, etc.
    - Continuous: Raw parameter values for T1, T2, FA, TI, etc.
    - Multi-hot: Multiple contrast attributes
    """
    
    def __init__(self, 
                 contrast_dim: int,           # Input dimension (e.g., 5 for 5 contrasts)
                 embedding_dim: int = 512,   # Output embedding dimension
                 num_contrasts: int = 5):    # Number of contrast types
        super().__init__()
        self.contrast_dim = contrast_dim
        self.embedding_dim = embedding_dim
        self.num_contrasts = num_contrasts
        
        # Option 1: Simple MLP for embedding
        self.mlp = nn.Sequential(
            nn.Linear(contrast_dim, 256),
            nn.SiLU(),
            nn.Linear(256, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Option 2: Separate embeddings for each contrast type
        self.contrast_embeddings = nn.Embedding(num_contrasts, embedding_dim)
        
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: [batch_size, contrast_dim] - contrast parameters
                   e.g., [batch, 5] where 5 is one-hot or continuous contrast params
        
        Returns:
            context: [batch_size, embedding_dim] - learnable embeddings
        """
        # Method 1: Pure MLP embedding
        context = self.mlp(theta)
        return context


class AdaptiveContrastEmbedding(nn.Module):
    """
    More sophisticated embedding that can handle multiple representation formats:
    - One-hot contrast labels
    - Continuous biophysical parameters (T1, T2 relaxation times, FA, etc.)
    - Mixed representations
    """
    
    def __init__(self,
                 contrast_dim: int,
                 embedding_dim: int = 512,
                 hidden_dim: int = 256):
        super().__init__()
        self.contrast_dim = contrast_dim
        self.embedding_dim = embedding_dim
        
        # Main embedding pathway
        self.encoder = nn.Sequential(
            nn.Linear(contrast_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Optional: Additional projection for spatialization
        # This can be useful if you want to expand [B, D] to [B, D, H, W]
        self.spatial_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: [B, contrast_dim]
        Returns:
            context: [B, embedding_dim] or [B, embedding_dim, 1, 1] if spatializing
        """
        context = self.encoder(theta)
        return context
    
    def to_spatial(self, context: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        Optionally expand context to spatial dimensions for feature injection.
        Args:
            context: [B, embedding_dim]
            spatial_shape: (H, W) target spatial dimensions
        Returns:
            spatial_context: [B, embedding_dim, H, W]
        """
        B, D = context.shape
        H, W = spatial_shape
        # Repeat across spatial dimensions
        spatial = context.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)
        return spatial