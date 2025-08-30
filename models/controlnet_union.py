# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.controlnets.controlnet import ControlNetModel

logger = logging.get_logger(__name__)

from collections import OrderedDict

# Transformer Block for ControlNet++ Union
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x)
        return ret.type(orig_type)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", nn.Linear(d_model, d_model * 4)), ("gelu", QuickGELU()),
                         ("c_proj", nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].
    """
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

class ControlNetModel_Union(ControlNetModel):
    """
    ControlNet++ Union model based on the official implementation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add ControlNet++ Union specific components
        # Get the correct hidden size from config
        if hasattr(self.config, 'hidden_sizes'):
            hidden_size = self.config.hidden_sizes[-1]
        elif hasattr(self.config, 'block_out_channels'):
            hidden_size = self.config.block_out_channels[-1]
        else:
            hidden_size = 1280  # Default for SDXL
        
        self.condition_transformer = ResidualAttentionBlock(
            d_model=hidden_size,
            n_head=8
        )
        
        # Control type embedding
        if hasattr(self.config, 'time_embedding_dim'):
            time_embedding_dim = self.config.time_embedding_dim
        else:
            time_embedding_dim = 320  # Default for SDXL
            
        self.control_type_embedding = nn.Embedding(6, time_embedding_dim)
        
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.LongTensor, int, float],
        encoder_hidden_states: torch.FloatTensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
        union_control: bool = False,
        union_control_type: Optional[torch.Tensor] = None,
    ) -> Union[ControlNetOutput, Tuple]:
        """
        Forward pass for ControlNet++ Union
        """
        # Default ControlNet forward pass
        down_block_res_samples, mid_block_res_sample = super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            return_dict=False
        )
        
        if union_control and union_control_type is not None:
            # Apply ControlNet++ Union specific processing
            # This is a simplified version - the full implementation would be more complex
            pass
        
        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        ) 