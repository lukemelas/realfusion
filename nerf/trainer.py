import glob
import os
import random
import time
from pathlib import Path
from typing import Optional, Mapping
from os.path import join as pjoin

import imageio
import numpy as np
import tensorboardX
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from rich.console import Console
from torch import Tensor
from torch_ema import ExponentialMovingAverage
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import make_grid

from nerf.options import Options, AnnealedValue
from nerf.renderer import NeRFRenderer
from nerf.utils import to_pil, safe_normalize
from sd.sd import StableDiffusion


class Trainer(object):
    def __init__(
        self, 
        name: str,  # name of this experiment
        opt: Options,  # extra conf
        model: NeRFRenderer,  # network 
        guidance: StableDiffusion,  # guidance network
        optimizer = None,  # optimizer
        ema_decay = None,  # if use EMA, set the decay
        lr_scheduler = None,  # scheduler
        metrics = [],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank = 0,  # which GPU am I
        world_size = 1,  # total num of GPUs
        device = None,  # device to use, usually setting to None is OK. (auto choose device)
        mute = False,  # whether to mute all print
        fp16 = False,  # amp optimize level
        eval_interval = 1,  # eval once every $ epoch
        max_keep_ckpt = 1,  # max num of saved ckpts in disk
        workspace = 'workspace',  # workspace to save logs & ckpts
        best_mode = 'min',  # the smaller/larger result, the better
        use_loss_as_metric = True,  # use loss as the first metric
        report_metric_at_train = False,  # also report metrics at training
        use_checkpoint = "latest",  # which ckpt to use at init time
        use_tensorboardX = True,  # whether to use tensorboard for logging
        scheduler_update_every_step = False,  # whether to call scheduler.step() after every train step
    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        
        # Model and guidance
        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model
        self.guidance = guidance

        # Text embedding
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
            self.prepare_text_embeddings()
        else:
            self.text_z = None

        # Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)
        
        # Scheduler
        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        
        # EMA
        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        # Loss scaling
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # Variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # Auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # Workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = pjoin(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")
            self.ckpt_path = pjoin(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
        
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] num parameters: {sum([p.numel() for p in model.parameters()]):_}')
        self.log(f'[INFO] num parameters w/ grad: {sum([p.numel() for p in model.parameters() if p.requires_grad]):_}')

        # Load checkpoint
        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # Flags
        self.use_synthetic_data = self.is_ever_nonzero(self.opt.lambda_prior)
        self.use_real_data = self.is_ever_nonzero(self.opt.lambda_image) or self.is_ever_nonzero(self.opt.lambda_mask)

    def is_ever_nonzero(self, loss_value: AnnealedValue):
        if isinstance(loss_value, float):
            return loss_value > 0
        elif len(loss_value) == 1:
            return loss_value[0] > 0
        elif len(loss_value) > 1:
            return loss_value[0] > 0 or loss_value[1] > 0  # either start or 

    def val(self, loss_value: AnnealedValue):
        """A helper function for getting a value that may be annealed over the course of training"""
        if isinstance(loss_value, float):
            value = loss_value
        elif len(loss_value) == 1:
            value = loss_value[0]
        elif len(loss_value) > 1:
            if len(loss_value) == 2:
                start_value, end_value = loss_value
                end_iters, fn_name = self.global_step, 'linear'  # default to linear over the course of training
            elif len(loss_value) == 4:
                start_value, end_value, end_iters, fn_name = loss_value
            else:
                raise ValueError()
            iter_fraction = min(end_iters / self.opt.iters, 1)
            if fn_name == 'linear':
                value = iter_fraction * end_value + (1 - iter_fraction) * start_value
            elif fn_name == 'log':
                value = end_value ** iter_fraction * start_value ** (1 - iter_fraction)
            else:
                raise ValueError(fn_name)
        else:
            raise ValueError(loss_value)
        return value

    def prepare_text_embeddings(self):
        """Compute text embeddings"""

        if self.opt.text is None:
            self.log("[WARN] text prompt is not provided.")
            self.text_z = None
            return

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                text = f"{self.opt.text}, {d} view"

                # Only front 
                negative_text = f"{self.opt.negative}"
                if self.opt.suppress_face:
                    if negative_text != '': 
                        negative_text += ', '
                    if d == 'back': 
                        negative_text += self.opt.suppress_face
                    elif d == 'side': 
                        negative_text += self.opt.suppress_face
                    elif d == 'overhead': 
                        negative_text += self.opt.suppress_face
                    elif d == 'bottom': 
                        negative_text += self.opt.suppress_face
                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

        print(f'Prepared text embeddings: {self.text_z}')

    def __del__(self):
        if hasattr(self, 'log_ptr'): 
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def debug_print(self, *args, **kwargs):
        if self.opt.debug:
            print(*args, **kwargs)

    def check_prompt(self):
        assert hasattr(self.guidance, '_sd')
        print('Checking stable diffusion model with prompt:')
        print(f'{self.opt.text}')
        print(self.guidance.tokenizer.tokenize(self.opt.text))
        # Generate
        image_check = self.guidance._sd.sample(prompt=[self.opt.text] * 3, guidance_scale=7.5)
        # Save
        output_dir_check = Path(self.workspace) / 'pretrain_check'
        output_dir_check.mkdir(exist_ok=True, parents=True)
        to_pil(image_check).save(output_dir_check / 'generations.png')
        (output_dir_check / 'prompt.txt').write_text(self.opt.text)
                
    def prepare_for_reconstruction(self, real_train_loader, real_train_full_image_loader, real_val_loader, real_test_loader):
        """Prepare optional real data loaders"""
        self.real_train_loader = real_train_loader
        self.real_train_iter = iter(self.real_train_loader)
        self.real_train_full_image_loader = real_train_full_image_loader
        self.real_val_loader = real_val_loader
        self.real_test_loader = real_test_loader

    def add_noise(self, data: Mapping[str, Tensor], noise: float = 0.0):
        # Note: For now, I'm just adding the same amount of data to the origins and directions. This
        # could certainly be done in a more sophisticated manner.
        if noise == 0: return data
        (B, N), device, dtype = data['rays_o'].shape[:2], data['rays_o'].device, data['rays_o'].dtype
        data['rays_o'] = data['rays_o'] + torch.randn(B, N, 3, device=device, dtype=dtype) * noise
        data['rays_d'] = data['rays_d'] + torch.randn(B, N, 3, device=device, dtype=dtype) * noise
        return data

    def get_real_data(self) -> dict:
        """Fetch a batch of real data"""
        try:
            data = next(self.real_train_iter)
        except StopIteration:
            self.real_train_iter = iter(self.real_train_loader)
            data = next(self.real_train_iter)
        return data

    def train_step_real_data(self):
        r"""Training step for real data"""

        # Unpack data
        data = self.get_real_data()
        rays_o: Tensor = data['rays_o'] # [B, N, 3]
        rays_d: Tensor = data['rays_d'] # [B, N, 3]
        gt_rgb: Tensor = data['images'][..., :3]  # [B, N, 3]
        gt_opacity: Tensor = data['images'][..., 3:]  # [B, N, 1] (alpha)
        B, N = rays_o.shape[:2]
        bg_color = 1

        # Add noise
        if self.opt.noise_real_camera > 0:
            camera_noise_factor = (1 - self.global_step / self.opt.iters) if self.opt.noise_real_camera_annealing else 1
            camera_noise = self.opt.noise_real_camera * camera_noise_factor
            rays_o = rays_o + torch.randn(B, 1, 3, device=rays_o.device, dtype=rays_o.dtype) * camera_noise
            rays_d = rays_d + torch.randn(B, 1, 3, device=rays_d.device, dtype=rays_d.dtype) * camera_noise

        # Render from view
        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, 
            ambient_ratio=1.0, shading='albedo', force_all_rays=False, **vars(self.opt))
        pred_rgb: torch.Tensor = outputs['image']  # [B, N, 3]
        pred_opacity: torch.Tensor = outputs['weights_sum'].reshape(B, N, 1)  # [B, N, 1]
        
        # Losses
        loss_real_dict = {}
        if self.val(self.opt.lambda_image) > 0:
            loss_real_image = self.val(self.opt.lambda_image) * F.mse_loss(pred_rgb * gt_opacity, gt_rgb * gt_opacity)
            loss_real_dict.update(dict(loss_real_image=loss_real_image))
        if self.val(self.opt.lambda_mask) > 0:  # mask view loss
            loss_real_mask = self.val(self.opt.lambda_mask) * F.mse_loss(pred_opacity, gt_opacity)
            loss_real_dict.update(dict(loss_real_mask=loss_real_mask))
        loss_real: Tensor = sum(loss_real_dict.values())

        return loss_real, loss_real_dict

    def visualize_train_step_real_data(self, name: Optional[str] = None):

        # Load data
        data = next(iter(self.real_train_full_image_loader))
        rays_o: Tensor = data['rays_o'] # [B, H, W, 3]
        rays_d: Tensor = data['rays_d'] # [B, H, W, 3]
        H, W = data['H'], data['W']
        gt_rgb: Tensor = data['images'][..., :3]  # [B, H, W, 3]
        gt_opacity: Tensor = data['images'][..., 3:].expand(-1, -1, -1, 3)  # [B, H, W, 1] (alpha)
        bg_color = 1

        # Render
        outputs: Mapping[str, Tensor] = self.model.render(
            rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, 
            ambient_ratio=1.0, shading='albedo', force_all_rays=True, **vars(self.opt)
        )

        # Unpack
        pred_rgb: torch.Tensor = outputs['image'].reshape(-1, H, W, 3)
        pred_opacity: torch.Tensor = outputs['weights_sum'].reshape(-1, H, W, 1).expand(-1, -1, -1, 3)
        pred_depth: torch.Tensor = outputs['depth'].reshape(-1, H, W, 1).expand(-1, -1, -1, 3).clamp(0, 1)
        vis_images = (gt_rgb, gt_opacity, gt_rgb * gt_opacity, pred_rgb, pred_opacity, pred_depth)

        # Save
        file_name = 'train_step_real_grid.png' if name is None else f'{name}_grid.png'
        output_dir = Path(self.workspace) / 'validation_real'
        output_dir.mkdir(exist_ok=True, parents=True)
        to_pil(torch.cat(vis_images).permute(0, 3, 1, 2)).save(output_dir / file_name)

        return vis_images

    def train_step_synthetic_data(self, data):

        # Unpack
        rays_o: Tensor = data['rays_o'] # [B, N, 3]
        rays_d: Tensor = data['rays_d'] # [B, N, 3]
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        bg_color = torch.rand((B * N, 3), device=rays_o.device)  # pixel-wise random

        # Replace the train camera position with the reconstruction camera position (plus noise)
        is_replaced_camera = (self.opt.replace_synthetic_camera_every > 0 and self.global_step % self.opt.replace_synthetic_camera_every)
        if is_replaced_camera:
            # First replace the camera
            data = next(iter(self.real_train_full_image_loader))
            # Then add noise to the replaced camera origin and rays
            camera_noise = self.opt.replace_synthetic_camera_noise
            if camera_noise > 0:
                rays_o = rays_o + torch.randn(B, 1, 3, device=rays_o.device, dtype=rays_o.dtype) * camera_noise
                rays_d = rays_d + torch.randn(B, 1, 3, device=rays_d.device, dtype=rays_d.dtype) * camera_noise

        # Shading
        if self.global_step < self.opt.albedo_iters:
            shading = 'albedo'
            ambient_ratio = 1.0
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
                ambient_ratio = 1.0
            elif rand > 0.4: 
                shading = 'textureless'
                ambient_ratio = 0.1
            else: 
                shading = 'lambertian'
                ambient_ratio = 0.1
    
        # Render
        outputs: Mapping[str, Tensor] = self.model.render(
            rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, 
            ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt)
        )

        # Reshape to image size
        pred = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
        pred_ws = outputs['weights_sum'].reshape(B, 1, H, W)

        # Save intermediate results during training steps
        if outputs['normals'] is not None and self.global_step % 100 == 0:
            output_dir_training = Path(self.workspace) / 'training_images'
            (output_dir_training / 'images').mkdir(exist_ok=True, parents=True)
            (output_dir_training / 'depth').mkdir(exist_ok=True, parents=True)
            (output_dir_training / 'normals').mkdir(exist_ok=True, parents=True)
            _pred = pred
            _depth = outputs['depth'].reshape(B, H, W).unsqueeze(1).contiguous()  # [1, 1, H, W]
            _normals = outputs['normals'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
            to_pil(_pred).save(output_dir_training / 'images' / f'images_{self.global_step:05d}.png')  # save
            to_pil(_depth).save(output_dir_training / 'depth' / f'depth_{self.global_step:05d}.png')  # save
            to_pil(_normals).save(output_dir_training / 'normals' / f'normals_{self.global_step:05d}.png')  # save
            print(f'\nSaved training images and normals at global step {self.global_step}')
        
        # Save debug information
        if self.opt.debug and self.global_step % 50 == 0:
            output_dir = Path(self.workspace) / 'debug'
            output_dir.mkdir(exist_ok=True, parents=True)
            to_pil(pred).save(output_dir / f'pred_{self.global_step:04d}_grid.png')
        
        # Text embeddings
        emb = self.text_z[data['dir']] if self.opt.dir_text and not is_replaced_camera else self.text_z[0]

        # Prior and regularization
        loss = self.guidance.train_step(
            emb, pred, lambda_grad=self.val(self.opt.lambda_prior), guidance_scale=self.opt.guidance_scale
        )

        if self.val(self.opt.lambda_opacity) > 0:
            loss_opacity = (pred_ws ** 2).mean()
            loss = loss + self.val(self.opt.lambda_opacity) * loss_opacity

        if self.val(self.opt.lambda_entropy) > 0:
            alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
            loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
            loss = loss + self.val(self.opt.lambda_entropy) * loss_entropy

        if self.val(self.opt.lambda_orient) > 0 and 'loss_orient' in outputs:
            loss_orient = outputs['loss_orient']
            loss = loss + self.val(self.opt.lambda_orient) * loss_orient

        if self.val(self.opt.lambda_smooth) > 0 and 'loss_smooth' in outputs:
            loss_smooth = outputs['loss_smooth']
            loss = loss + self.val(self.opt.lambda_smooth) * loss_smooth
        
        if self.val(self.opt.lambda_smooth_2d) > 0 and 'normals' in outputs:
            pred_normals = outputs['normals'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous() # [1, 3, H, W]
            pred_normals_blur = gaussian_blur(pred_normals, kernel_size=9).detach()
            loss_smooth_2d = F.mse_loss(pred_normals, pred_normals_blur)
            loss = loss + self.val(self.opt.lambda_smooth_2d) * loss_smooth_2d

        return loss

    def eval_step(self, data, bg_color: Optional[Tensor] = None, perturb: bool = False):

        # Unpack
        rays_o: Tensor = data['rays_o'] # [B, N, 3]
        rays_d: Tensor = data['rays_d'] # [B, N, 3]
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']
        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)

        # Render
        outputs: Mapping[str, Tensor] = self.model.render(
            rays_o, rays_d, staged=True, perturb=perturb, bg_color=bg_color, light_d=light_d, 
            ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True, **vars(self.opt)
        )
        
        # Reshape
        pred_rgb: Tensor = outputs['image'].reshape(B, H, W, 3).contiguous().clamp(0, 1)
        pred_opacity: Tensor = outputs['weights_sum'].reshape(B, H, W, 1)
        pred_depth: Tensor = outputs['depth'].reshape(B, H, W, 1)
        pred_normals: Tensor = outputs['normals'].reshape(B, H, W, 3) if 'normals' in outputs else None

        # Dummy value for the loss
        loss = torch.zeros_like(rays_o).sum().detach()

        return pred_rgb, pred_opacity, pred_depth, pred_normals, loss

    def test_step(self, data, bg_color=None, perturb=False):
        return self.eval_step(data, bg_color=bg_color, perturb=perturb)
    
    def train(self, train_loader, valid_loader, vis_loader_1, max_epochs):
        
        # Logging
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(pjoin(self.workspace, "run", self.name))

        # Loop
        start_t = time.time()
        start_e = time.time()
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.log(f"Training epoch took {(time.time() - start_e)/ 60:.4f} minutes (total: {(time.time() - start_t)/ 60:.4f}).")
            self.epoch = epoch

            # Train
            self.train_one_epoch(train_loader)

            # Eval and save 
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                if self.local_rank == 0:
                    self.save_checkpoint(full=False, best=True)  # for evaluation
        
        # End
        end_t = time.time()
        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    @torch.no_grad()
    def test(self, loader, save_path=None, name=None, write_video=True):
        self.model.eval()

        # Setup
        if save_path is None:
            save_path = pjoin(self.workspace, 'results')
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        Path(save_path).mkdir(exist_ok=True, parents=True)
        self.log(f"==> Start Test, save results to {save_path}")

        # For reconstruction
        if self.use_real_data and self.opt.test_on_real_data:
            loader = self.real_test_loader

        # Loop
        all_preds = []
        pbar_format = '{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        pbar = tqdm.tqdm(total=(len(loader) * loader.batch_size), bar_format=pbar_format)
        for i, data in enumerate(loader):
            
            # NOTE: The code below all assumes batch size is 1, which is fine for now

            # Test setup
            with torch.cuda.amp.autocast(enabled=self.fp16):
                _kwargs = {'ambient_ratio': 0.1, 'light_d': safe_normalize(data['rays_o'][0][0])}
                pred_textureless, _, _, _, _ = self.test_step({**data, **_kwargs, 'shading': 'textureless'})
                pred_rgb, pred_opacity, pred_depth, pred_normals, loss = self.test_step(data)
                pred_opacity = pred_opacity.expand(-1, -1, -1, 3).clip(0, 1)  # (B, H, W, 3)
                pred_depth = pred_depth.expand(-1, -1, -1, 3).clip(0, 1)  # (B, H, W, 3)
            
            # Concat
            preds = [pred_rgb, pred_opacity, pred_depth, pred_normals, pred_textureless]
            if self.use_real_data:
                gt_rgb: Tensor = next(iter(self.real_train_full_image_loader))['images'][..., :3]  # [B, H, W, 3]
                gt_rgb: Tensor = F.interpolate(gt_rgb.permute(0, 3, 1, 2), size=pred_rgb.shape[1:3], mode='bilinear').permute(0, 2, 3, 1)
            else:
                gt_rgb = torch.zeros_like(pred_rgb)
            preds.insert(0, gt_rgb)
            all_preds.append(preds)  # all are (B, H, W, 3), B = 1

            # Progress bar
            pbar.update(loader.batch_size)
        
        # Write frames
        if self.use_real_data and self.opt.test_on_real_data:
            subdir = f'{self.opt.blender_test_split}_frames' if self.opt.real_dataset_type == 'blender' else 'test_frames'
            frames_dir = Path(save_path).parent / subdir
            frames_dir.mkdir(exist_ok=True, parents=True)
            for i, preds in enumerate(all_preds):
                pred_rgb = preds[1].permute(0, 3, 1, 2)  # (B, 3, H, W)
                to_pil(pred_rgb, padding=0).save(frames_dir / f'test_{i:03d}.png')

        # Visualize video
        if write_video:

            # Separate preds
            preds_rgb = torch.cat([preds[1] for preds in all_preds])   # (F, H, W, 3)
            preds_opacity = torch.cat([preds[2] for preds in all_preds])   # (F, H, W, 3)
            preds_depth = torch.cat([preds[3] for preds in all_preds])   # (F, H, W, 3)
            preds_normals = torch.cat([preds[4] for preds in all_preds])   # (F, H, W, 3)
            preds_textureless = torch.cat([preds[5] for preds in all_preds])   # (F, H, W, 3)
            preds_grid = torch.stack([make_grid(torch.cat(preds).permute(0, 3, 1, 2), padding=0, nrow=3) 
                                      for preds in all_preds]).permute(0, 2, 3, 1)  # (F, H*2, W*3, 3)
            
            # Save videos
            preds_tensors = preds_rgb, preds_opacity, preds_depth, preds_normals, preds_textureless, preds_grid
            render_names = ('rgb', 'opacity', 'depth', 'normals', 'textureless', 'grid')
            for render_name, preds_tensor in zip(render_names, preds_tensors):
                print(render_name)
                preds_np = (preds_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
                imageio.mimwrite(pjoin(save_path, f'{name}_{render_name}.mp4'), preds_np,
                    fps=25, quality=8, macro_block_size=1)

        self.log("==> Finished Test.")

    def train_one_epoch(self, loader):
        self.model.train()
        
        # Setup
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)        
        if self.local_rank == 0:
            pbar_format = '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format=pbar_format)

        # Loop
        self.local_step = 0
        for data in loader:
            self.local_step += 1
            self.global_step += 1

            # Update grid bitfield every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            # Update grid
            if (getattr(self.model, 'grid_levels_mask', 0) > 0) and (self.global_step > self.opt.grid_levels_mask_iters):
                print(f'Increasing grid levels from {self.opt.grid_levels - self.opt.grid_levels_mask} to {self.opt.grid_levels}')
                self.model.grid_levels_mask = 0
                    
            # Loss
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                
                # Train step: synthetic data
                if self.use_synthetic_data > 0:
                    loss_synthetic = self.train_step_synthetic_data(data)
                else:
                    loss_synthetic = 0
            
                # Train step: real data
                if self.use_real_data and self.global_step >= self.opt.real_iters and self.global_step % self.opt.real_every == 0:
                    loss_real, loss_real_dict = self.train_step_real_data()
                else:
                    loss_real, loss_real_dict = 0, {}
            
            # Backward
            loss = loss_synthetic + loss_real
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            # Log
            loss_val = loss.item()
            total_loss += loss_val
            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)
                loss_real_str = ' '.join([f'{k}: {(v.item() if torch.is_tensor(v) else v):.4f}' for k, v in loss_real_dict.items()])
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f} {loss_real_str}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}) {loss_real_str}")
                pbar.update(loader.batch_size)

        # Update
        if self.ema is not None:
            self.ema.update()
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    @torch.no_grad()
    def evaluate_one_epoch(self, loader, name=None):
        self.model.eval()
        
        # Setup
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        # For reconstruction
        if self.use_real_data:
            
            # Visualize and save reconstruction training step
            self.visualize_train_step_real_data(name=name)  # 6 tensors of shape (B, H, W, 3) for B = 1

            # Use the real val loader, not the synthetic one
            loader = self.real_val_loader

        # EMA
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        # Progress bar
        if self.local_rank == 0:
            pbar_format = '{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format=pbar_format)
        
        self.local_step = 0
        all_preds = []
        for data in loader:    
            self.local_step += 1

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgb, pred_opacity, pred_depth, pred_normals, loss = self.eval_step(data)

            if self.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / self.world_size
                
                preds_list = [torch.zeros_like(pred_rgb).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(preds_list, pred_rgb)
                pred_rgb = torch.cat(preds_list, dim=0)
                
                preds_list = [torch.zeros_like(pred_opacity).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(preds_list, pred_opacity)
                pred_opacity = torch.cat(preds_list, dim=0)
                
                preds_list = [torch.zeros_like(pred_depth).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(preds_list, pred_depth)
                pred_depth = torch.cat(preds_list, dim=0)
                
                preds_list = [torch.zeros_like(pred_normals).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(preds_list, pred_normals)
                pred_normals = torch.cat(preds_list, dim=0)
            
            loss_val = loss.item()
            total_loss += loss_val

            # New: saving an image grid instead of individual images for ease of visualization
            pred_opacity = pred_opacity.expand(-1, -1, -1, 3).clip(0, 1)  # (B, H, W, 3)
            pred_depth = pred_depth.expand(-1, -1, -1, 3).clip(0, 1)  # (B, H, W, 3)
            all_preds.append(torch.cat((pred_rgb, pred_opacity, pred_depth)))  # (3*B, H, W, 3)  # excluding normals for image size

            # Evaluate if using real data
            if self.use_real_data and self.local_rank == 0:
                for metric in self.metrics:
                    print(f"{pred_rgb=}")
                    print(f"{data['images']=}")
                    metric.update(pred_rgb, data['images'])

        # Save grids instead of individual images during validation
        output_dir = Path(self.workspace) / 'validation_synthetic'
        output_dir.mkdir(exist_ok=True, parents=True)
        vis_grid = torch.stack(all_preds[-8:]).transpose(0, 1).flatten(0, 1)  # (3*B*F, H, W, 3)
        to_pil(vis_grid.permute(0, 3, 1, 2), nrow=len(vis_grid) // 3).save(output_dir / f'{name}_grid.png')

        # Log
        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)
        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        # EMA
        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = pjoin(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, pjoin(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                
                # Save ema results 
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                
                # Get state dict
                state['model'] = self.model.state_dict()

                # Restore ema
                if self.ema is not None:
                    self.ema.restore()
                
                torch.save(state, self.best_path)
            else:
                self.log("[WARN] no evaluated results found, skip saving best checkpoint.")
        
        torch.save(state, pjoin(self.ckpt_path, 'current.pth'))
                
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except Exception:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except Exception:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except Exception:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except Exception:
                self.log("[WARN] Failed to load scaler.")

    def save_mesh(self, save_path=None, resolution=128):
        if save_path is None:
            save_path = pjoin(self.workspace, 'mesh')
        self.log(f"==> Saving mesh to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        self.model.export_mesh(save_path, resolution=resolution)
        self.log("==> Finished saving mesh.")
