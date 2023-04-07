import inspect
import random
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_ancestral_discrete import \
    EulerAncestralDiscreteScheduler
from diffusers.schedulers.scheduling_euler_discrete import \
    EulerDiscreteScheduler
from diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from diffusers.schedulers.scheduling_ipndm import IPNDMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


SCHEDULER_MAP = {
    'ddpm': DDPMScheduler, 
    'ddim': DDIMScheduler, 
    'pndm': PNDMScheduler, 
    'ipndm': IPNDMScheduler, 
    'euler': EulerDiscreteScheduler, 
    'euler_anc': EulerAncestralDiscreteScheduler, 
    'heun': HeunDiscreteScheduler, 
    'dpmsolver': DPMSolverMultistepScheduler,
}


def default(x, d):
    return d if x is None else x


class StableDiffusionModel(DiffusionPipeline, torch.nn.Module):
    """
    This is a wrapper class. It was designed for an old version of the diffusers library. Fortunately, 
    recent version of diffusers has addressed most of the things that this was designed to patch. :)
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[SchedulerMixin, ConfigMixin],
    ):
        super().__init__()
        
        # Register modules and declare type hints
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler
        )
        self.vae: AutoencoderKL
        self.text_encoder: CLIPTextModel
        self.tokenizer: CLIPTokenizer
        self.unet: UNet2DConditionModel
        self.scheduler: Union[SchedulerMixin, ConfigMixin]

        # Scheduler for training (can be set manually if desired)
        self.train_scheduler = DDPMScheduler.from_config(self.scheduler.config)

        # Additional helpful properties
        self.tokenizer_kwargs = dict(truncation=True, padding="max_length", 
            max_length=self.tokenizer.model_max_length, return_tensors="pt")

        # Disable gradient for all models except unet (which is kept as default)
        self.set_requires_grad(False, exclude=[self.unet])

        # Whether to cache the output of the text encoder
        self.use_cache_text_encoder = False  # soon will switch to True

    def get_scheduler(self, scheduler: Union[str, SchedulerMixin]) -> SchedulerMixin:
        """Get scheduler if passed as string"""
        if isinstance(scheduler, str):
            scheduler = SCHEDULER_MAP.get(scheduler).from_config(self.scheduler.config)
        if not isinstance(scheduler, SchedulerMixin):
            raise ValueError(f'Expected subclass of SchedulerMixin but got: {scheduler}')
        return scheduler

    def set_requires_grad(self, requires_grad: bool, exclude: Sequence = []):
        for name, param in self.named_parameters():
            if param not in exclude:
                param.requires_grad_(requires_grad)

    def prepare_prompt(self, prompt: Optional[Tuple[str]] = None, batch_size: int = 1, text_dropout_prob: float = 0.0):
        """Format prompt for tokenizer"""
        if torch.is_tensor(prompt) and torch.is_floating_point(prompt) and len(prompt.shape) == 3:
            return prompt  # prompt is already text embedding
        if text_dropout_prob > 0 and random.random() < text_dropout_prob:
            prompt = ''  # for classifier-free guidance aware training
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size  # 
        if not isinstance(prompt, Sequence):
            raise ValueError(f'Unexpected prompt {prompt} of type {type(prompt)}')
        return tuple(prompt)

    def _encode_text(self, prompt: Tuple[str]) -> Tensor:
        """A helper function for encoding a prompt with the tokenizer and text encoder"""
        text_input = self.tokenizer(prompt, **self.tokenizer_kwargs)
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]  # [0] gets hidden state
        return text_embeddings

    @lru_cache(maxsize=8)
    def _encode_text_with_caching(self, prompt: Tuple[str]):
        """A helper function for caching the output of the text encoder"""
        return self.encode_text(prompt)

    def encode_text(self, prompt: Optional[Tuple[str]] = None, batch_size: int = 1, text_dropout_prob: float = 0.0):
        """Tokenizes and encodes text with the text encoder model"""
        prompt: Tuple[str] = self.prepare_prompt(prompt, batch_size=batch_size, text_dropout_prob=text_dropout_prob)
        return self._encode_text_with_caching(prompt) if self.use_cache_text_encoder else self._encode_text(prompt)

    def encode_image(self, inputs: Tensor, scale: float = 0.18215, generator: Optional[torch.Generator] = None):
        """Runs image through VAE encoder"""
        inputs = 2.0 * inputs - 1.0  # image normalization
        x_0 = self.vae.encode(inputs).latent_dist.sample(generator=generator)  # (B, C_lat, H_lat, W_lat)
        x_0 = x_0 * scale
        return x_0

    def decode_image(self, latents: Tensor, scale: float = 0.18215, clamp: bool = True):
        """Runs latents through VAE decoder"""
        latents = latents / scale
        image = self.vae.decode(latents).sample
        image = image * 0.5 + 0.5  # undo image normalization
        return torch.clamp(image, 0, 1) if clamp else image

    def get_target(self, x_0: Tensor, scheduler: DDPMScheduler, noise: Optional[Tensor] = None, 
            timesteps: Optional[Tensor] = None):
        """Get the target for loss depending on the prediction type"""
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(x_0, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")
        return target

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: Optional[int] = None, 
        prompt: Optional[List[str]] = None,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 0.0,
        negative_prompt: str = '',
        eta: Optional[float] = 0.0,
        latents: Optional[torch.FloatTensor] = None,
        scheduler: Union[SchedulerMixin, str] = 'ddim',
        return_latents: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        
        # Get batch size
        if batch_size is not None:
            B = batch_size
        elif prompt is not None:
            B = batch_size = 1 if isinstance(prompt, str) else len(prompt)
        else:
            raise NotImplementedError()
        
        # Get scheduler for sampling
        scheduler = self.get_scheduler(scheduler)

        # Check
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        H, W, H_lat, W_lat = height, width, height // 8, width // 8
        C_lat = self.unet.in_channels

        # Encode text
        text_embeddings = self.encode_text(prompt, batch_size=B)

        # Classifier-free guidance
        do_classifier_free_guidance = (guidance_scale > 1.0)
        if do_classifier_free_guidance:
            negative_embeddings = self.encode_text(negative_prompt, batch_size=B)
            text_embeddings = torch.cat([negative_embeddings, text_embeddings])

        # Sample noise
        latents_shape = (B, C_lat, H_lat, W_lat)
        if latents is None:
            latents = torch.randn(latents_shape, device=self.device, generator=generator, dtype=text_embeddings.dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(device=self.device, dtype=text_embeddings.dtype)

        # Set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Loop over timesteps
        for i, t in (enumerate(tqdm(scheduler.timesteps, desc=f'sampling size {latents_shape}'))):

            # Scale latents if using linear multistep scheduler
            latent_model_input = latents

            # Expand the latent_model_input if we are doing classifier free guidance
            latent_model_input = torch.cat([latent_model_input] * 2) if do_classifier_free_guidance else latent_model_input
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Classifier guidance
            if do_classifier_free_guidance:
                noise_pred_baseline, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_baseline + guidance_scale * (noise_pred_text - noise_pred_baseline)
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # Scale and decode the image latents with vae
        output = latents if return_latents else self.decode_image(latents, clamp=True)

        return output


