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

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.utils.import_utils import is_invisible_watermark_available
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, StableDiffusionXLLoraLoaderMixin, TextualInversionLoaderMixin, IPAdapterMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel, ImageProjection
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
        >>> negative_prompt = "low quality, bad quality, sketches"

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
        ... )

        >>> # initialize the models and pipeline
        >>> controlnet_conditioning_scale = 0.5  # recommended for good generalization
        >>> controlnet = ControlNetModel.from_pretrained(
        ...     "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ... )
        >>> vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> # get canny image
        >>> image = np.array(image)
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # generate image
        >>> image = pipe(
        ...     prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
        ... ).images[0]
        ```
"""

class StableDiffusionXLControlNetUnionPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using Stable Diffusion XL with ControlNet++ Union.
    """
    
    model_cpu_offload_seq = "TextEncoder->TextEncoder2->UNet->ControlNet->VAE"
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel]],
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()
        
        # Register components
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        
        # Add watermarker
        if add_watermarker is None:
            add_watermarker = is_invisible_watermark_available()
        
        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None
            
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        image_list: Optional[List] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        union_control: bool = False,
        union_control_type: Optional[torch.Tensor] = None,
    ) -> Union[StableDiffusionXLPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        latents_device = torch.device("cpu") if self.scheduler.config.force_zeros_for_empty_prompt else device
        
        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        )
        
        # 4. Prepare controlnet condition
        if image is not None:
            if isinstance(image, PIL.Image.Image):
                image = [image]
            
            if image_list is not None and union_control:
                # Use image_list for ControlNet++ Union
                controlnet_cond = image_list
            else:
                # Use single image for regular ControlNet
                controlnet_cond = image
        else:
            controlnet_cond = None
            
        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        # 7. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)
        
        # 8. Add image to the conditioning
        if controlnet_cond is not None:
            if union_control and union_control_type is not None:
                # ControlNet++ Union processing
                controlnet_cond = self.prepare_controlnet_union_condition(
                    controlnet_cond, union_control_type
                )
            else:
                # Regular ControlNet processing
                controlnet_cond = self.prepare_controlnet_condition(controlnet_cond)
        
        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # ControlNet prediction
                if controlnet_cond is not None:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=controlnet_conditioning_scale,
                        return_dict=False,
                        union_control=union_control,
                        union_control_type=union_control_type,
                    )
                else:
                    down_block_res_samples, mid_block_res_sample = None, None
                
                # Predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                # Call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # 10. Post-processing
        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # 11. Offload all models
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return (image,)
            
        return StableDiffusionXLPipelineOutput(images=image)
    
    def prepare_controlnet_union_condition(self, image_list, union_control_type):
        """
        Prepare ControlNet++ Union condition
        """
        # This is a simplified implementation
        # The full implementation would handle multiple control types
        return image_list
    
    def prepare_controlnet_condition(self, image):
        """
        Prepare regular ControlNet condition
        """
        if isinstance(image, list):
            image = [self.image_processor.preprocess(img) for img in image]
        else:
            image = self.image_processor.preprocess(image)
        return image 