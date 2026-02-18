import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm

from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()
        self.vqgan.train = disabled_train
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond, context=None):
        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        context = self.get_cond_stage_context(x_cond)
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.decode(x_latent, cond=False)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out

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
    
    def __init__(self, model_config):
        super().__init__(model_config)
        
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
        
        return self.p_losses(x_latent, theta, context, t)
    
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