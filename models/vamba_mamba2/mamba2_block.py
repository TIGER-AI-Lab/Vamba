# modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/block.py, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py
import math
import torch
from torch import nn, Tensor

import time
import easydict

from mamba_ssm.ops.triton.layer_norm import RMSNorm

from .mamba2_mixer import Mamba2

def _init_weights(
    module,
    n_layer,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Mamba2Block(nn.Module):
    def __init__(
        self, config, layer_idx):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        norm_type = config.norm_type
        if norm_type == "rmsnorm":
            norm_cls = RMSNorm
        elif norm_type == "layernorm":
            norm_cls = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        hidden_dim = config.hidden_dim
        self.norm = norm_cls(hidden_dim)
        self.mixer = Mamba2(
            config=config,
            layer_idx=str(layer_idx),
        )
        self.layer_idx = str(layer_idx)

    def forward(self, hidden_states: Tensor, inference_params=None, **mixer_kwargs):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        
        # if self.config.sequential_inference and not self.training:
        if False:
            batch, length, _ = hidden_states.shape
            inference_params = easydict.EasyDict()
            inference_params.key_value_memory_dict = {self.layer_idx: self.allocate_inference_cache(batch, max_seqlen=None)}
            inference_params.seqlen_offset = 0
            
            all_hidden_states = []
            chunk_size = length // 16

            for i in range(chunk_size):
                y_ = self.mixer(hidden_states[:, i*(length//chunk_size):(i+1)*(length//chunk_size)], inference_params=inference_params, **mixer_kwargs)
                all_hidden_states.append(y_)
            hidden_states = torch.cat(all_hidden_states, dim=1)
                
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        hidden_states = residual + hidden_states
        return hidden_states

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)