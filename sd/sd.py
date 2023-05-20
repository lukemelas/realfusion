import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from sd.wrapper import StableDiffusionModel


class StableDiffusion(nn.Module):
    def __init__(
        self, 
        stable_diffusion_model: StableDiffusionModel,
        device: torch.device, 
    ):
        super().__init__()
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        # Load diffusion model
        self._sd = stable_diffusion_model.to(device)
                
        # Components
        self.vae: AutoencoderKL = self._sd.vae
        self.tokenizer: CLIPTokenizer = self._sd.tokenizer
        self.text_encoder: CLIPTextModel = self._sd.text_encoder
        self.unet: UNet2DConditionModel = self._sd.unet
        self.scheduler: PNDMScheduler = self._sd.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        
        # Tokenizer kwargs
        kwargs = dict(padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        # Positive embeddings
        pos_input: torch.Tensor = self.tokenizer(prompt, **kwargs)
        pos_embeddings = self.text_encoder(pos_input.input_ids.to(self.device))[0]

        # Negative embeddings
        neg_input: torch.Tensor = self.tokenizer(negative_prompt, **kwargs)
        neg_embeddings = self.text_encoder(neg_input.input_ids.to(self.device))[0]

        return torch.cat([neg_embeddings, pos_embeddings])

    def train_step(self, emb, pred_rgb, noise=None, guidance_scale=100, lambda_grad=1.0, mask_grad=None, steps=1):
        
        # Encode image
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512)

        # Sample timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # Predict the noise residual with unet and classifier-free guidance. NO GRAD!
        with torch.no_grad():

            # Sample noise
            if noise is None:
                noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # CFG
            latent_model_input = torch.cat([latents_noisy] * 2)

            # Forward
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=emb).sample

            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Weight function. Can also use `w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])` or maybe `w = 1`
            w = (1 - self.alphas[t])
            grad = lambda_grad * w * (noise_pred - noise)
        
            # Mask
            if mask_grad is not None:
                mask_grad = F.interpolate(mask_grad, (64, 64), mode='bilinear', align_corners=False)
                grad = grad * mask_grad

            # (Not sure if necessary) clip grad for stable training? Maybe `grad = grad.clamp(-10, 10)` ?
            grad = torch.nan_to_num(grad)

        # Manually backward, since we omitted an item in grad and cannot simply autodiff.
        latents.backward(gradient=grad, retain_graph=True)

        # Return something to represent the loss
        pseudo_loss = grad.abs().mean().detach()
        return pseudo_loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, 
            guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), 
                device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        imgs = 2 * imgs - 1  # imgs: [B, 3, H, W]
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, 
            guidance_scale=7.5, latents=None):

        # Setup
        if isinstance(prompts, str):
            prompts = [prompts]    
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs

        
if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
