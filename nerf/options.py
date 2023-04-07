import datetime
from pathlib import Path
from typing import Literal, Optional

from tap import Tap


# An AnnealedValue is a helper type for a value that may be annealed over the course of 
# training. It can either be a single value fixed for all of training, or a list of 
# [start_value, end_value] which is annealed linearly over all training iterations, or a 
# list of [start_value, end_value, end_iters, fn_name] which reaches its end_value at 
# end_iters and may use either linear or log annealing.
AnnealedValue = list[float]


class Options(Tap):

    # High-level
    O: bool = False  # equals --cuda_ray --dir_text
    O2: bool = False  # equals --backbone vanilla --dir_text
    test: bool = False

    # General
    seed: int = 101  # random seed
    run_name: str = 'default'  # (optional) tag a specific run
    workspace: Optional[str] = None  # (optional) specify an exact output directory

    # Prompt
    text: str = "A high-resolution DSLR image of a <token>"  # positive prompt
    negative: str = ''  # negative prompt

    # Stable diffusion model
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"  # HF model name
    pretrained_model_image_size: int = 512  # corresponding image size
    learned_embeds_path: Optional[str] = './examples/natural-images/fish_real_nemo/learned_embeds.bin'  # path to saved embeds dict, usually named learned_embeds.bin

    # Classifier-free guidance scale for score distillation sampling loss. Large values
    # seem to be necessary for good results, both in DreamFusion and here.
    guidance_scale: float = 100.0

    # Real dataset
    image_path: Optional[str] = './examples/natural-images/fish_real_nemo/rgba.png'  # path to real RGBA image for reconstruction

    # Synthetic dataset
    uniform_sphere_rate: float = 0.5  # likelihood of sampling camera location uniformly on the sphere surface area
    bound: float = 0.75  # assume the scene is bounded in box(-bound, bound)")  # NOTE: Previous bound was 1, but I think 0.75 is better
    dt_gamma: float = 0  # dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)
    min_near: float = 0.1  # minimum near distance for camera
    radius_range: tuple[float, float] = [1.0, 1.5]  # training camera radius range
    radius_rot: Optional[float] = 1.8  # None  # circle radius for vis
    fovy_range: tuple[float, float] = [40, 70]  # training camera fovy range
    dir_text: bool = True  # direction-encode the text prompt, by appending front/side/back/overhead view
    suppress_face: Optional[str] = None  # text for negative prompt for back view of image
    angle_overhead: float = 30  # [0, angle_overhead] is the overhead region
    angle_front: float = 60  # [0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.
    pose_angle: float = 75  # angle for visualization and reconstruction, note that it's 90-X degrees, not X degrees
    jitter_pose: bool = False  # add jitters to the randomly sampled camera poses
    
    # Training
    ckpt: str = 'latest'
    eval_interval: int = 5  # evaluate on the valid set every interval epochs
    iters: int = 5000  # training iters
    lr_warmup: Optional[bool] = None  # whether to use a learning rate warmup
    lr: float = 1e-3  # initial learning rate (only if using lr_warmup)
    min_lr: float = 1e-6  # minimal learning rate (only if using lr_warmup)
    warm_iters: int = 2000  # warmup iters
    cuda_ray: bool = True  # use CUDA raymarching instead of pytorch
    max_steps: int = 512  # max num steps sampled per ray (only valid when using --cuda_ray)
    num_steps: int = 64  # num steps sampled per ray (only valid when not using --cuda_ray)
    upsample_steps: int = 32  # num steps up-sampled per ray (only valid when not using --cuda_ray)
    update_extra_interval: int = 16  # iter interval to update extra status (only valid when using --cuda_ray)
    max_ray_batch: int = 4096  # batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)
    albedo: bool = False  # only use albedo shading to train, overrides --albedo_iters
    albedo_iters: int = 1000  # training iters that only use albedo shading
    num_rays: int = 4096  # (real data only) num rays sampled per image for each training step

    # Render sizes
    # note: at some point HW_real should replace --downsample for blender images
    HW_synthetic: int = 96  # render size for synthetic images
    HW_real: int = 128  # render size for real image, for image_only dataset
    HW_vis: int = 128  # render size for visualization (i.e. val, test)
    
    # Model
    backbone: Literal['grid'] = 'grid'  # nerf backbone
    bg_radius: float = 1.4  # use a background model at sphere (note: radius does not matter as long as > 0)
    density_thresh: float = 10  # threshold for density grid to be occupied
    blob_density: float = 5  # max (center) density for the gaussian density blob
    blob_radius: float = 0.2  # control the radius for the gaussian density blob
    grid_levels: int = 16  # the number of levels in the feature grid
    grid_resolution: int = 2048  #  maximum resolution of the feature grid
    grid_levels_mask: int = 8  # the number of levels in the feature grid to mask (to disable use 0)
    grid_levels_mask_iters: int = 3_000  # the number of iterations for feature grid masking (to disable use 1_000_000)
    optim: Literal['adan', 'adam', 'adamw'] = 'adamw'
    fp16: bool = False  # use amp mixed precision training
    ema_decay: float = 0.95  # exponential moving average of model weights

    # Loss and regularization
    lambda_prior: AnnealedValue = [1.0]  # loss scale for diffusion model
    lambda_image: AnnealedValue = [5.0]  # loss scale for real image
    lambda_mask: AnnealedValue = [0.5]  # loss scale for real mask
    lambda_entropy: AnnealedValue = [1e-4]  # loss scale for alpha entropy
    lambda_opacity: AnnealedValue = [0]  # loss scale for alpha value
    lambda_orient: AnnealedValue = [1e-2]  # loss scale for orientation
    lambda_smooth: AnnealedValue = [0]  # loss scale for surface smoothness
    lambda_smooth_2d: AnnealedValue = [0.5]  # loss scale for surface smoothness (2d version)

    # Reconstruction
    real_iters: int = 0  # start reconstructions iterations after X iterations of prior-only optimization
    real_every: int = 1  # do reconstruction every X iterations to save on compute
    replace_synthetic_camera_every: int = 10  # use the real camera in place of the synthetic camera every X steps
    replace_synthetic_camera_noise: float = 0.02  # std of noise to add to the real camera when used in place of the synthetic cam
    noise_real_camera: float = 0.001  # add noise to the reconstruction step
    noise_real_camera_annealing: bool = True  # anneal the noise to zero over the coarse of training
    
    # Misc
    save_mesh: bool = False  # export an obj mesh with texture
    save_test_name: str = 'df_test'  # identifier for saving test visualizations
    test_on_real_data: bool = False  # whether to do the test on real data or not
    wandb: bool = False  # Weights & Biases
    debug: bool = False  # debug mode (prints more, saves less)

    def process_args(self):

        # Acceleration
        if self.O:
            self.fp16 = True
            self.dir_text = True
            self.cuda_ray = True
        elif self.O2:
            self.fp16 = self.albedo
            self.dir_text = True
            self.backbone = 'vanilla'
        
        # Defaults
        if self.albedo:
            self.albedo_iters = self.iters
        if self.lr_warmup is None:
            self.lr_warmup = False  # (self.backbone == 'vanilla') when we add the vanilla option later on

        # Checks
        if (self.lambda_prior == 0) and (self.real_iters > 0 or self.real_every > 1): 
            raise ValueError('What are you doing?')
        
        # Debug
        if self.debug:
            # self.run_name = "debug"
            self.wandb = False
            self.workspace = None
        
        # Set up automatic token replacement for prompt
        if '<token>' in self.text or '<token>' in self.negative:
            if self.learned_embeds_path is None:
                raise ValueError('--learned_embeds_path must be specified when using <token>') 
            import torch
            tmp = list(torch.load(self.learned_embeds_path, map_location='cpu').keys())
            if len(tmp) != 1:
                raise ValueError('Something is wrong with the dict passed in for --learned_embeds_path') 
            token = tmp[0]
            self.text = self.text.replace('<token>', token)
            self.negative = self.negative.replace('<token>', token)
            print(f'Prompt after replacing <token>: {self.text}')
            print(f'Negative prompt after replacing <token>: {self.negative}')

        # Set up output directory
        if self.workspace is None:
            timestr = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
            self.workspace = f'outputs/{self.run_name}/{timestr}--seed-{self.seed}'
        Path(self.workspace).mkdir(exist_ok=True, parents=True)
        print(f'Workspace: {self.workspace}')

