import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from nerf.provider_image import NeRFDataset as ImageOnlyNeRFDataset
from nerf.provider_synthetic import NeRFDataset as SyntheticNeRFDataset
from nerf.utils import seed_everything, setup_distributed_print
from nerf.trainer import Trainer
from nerf.options import Options
from sd import StableDiffusion, StableDiffusionModel, add_tokens_to_model_from_path

def main():

    # I love lovely_tensors
    import lovely_tensors
    lovely_tensors.monkey_patch()

    # Arguments
    opt = Options().parse_args()
    seed_everything(opt.seed)
    print(opt)
    Path(opt.workspace).mkdir(exist_ok=(opt.ckpt != 'scratch'), parents=True)
    opt.save(str(Path(opt.workspace) / 'config.json'))
    (Path(opt.workspace) / 'command.txt').write_text(subprocess.list2cmdline(sys.argv[1:]))

    # Save and print config
    setup_distributed_print(True)

    # Create model
    if opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')
    model = NeRFNetwork(opt)    
    print(model)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create loaders for synthetic data
    train_loader = SyntheticNeRFDataset(
        opt, device=device, type='train', H=opt.HW_synthetic, W=opt.HW_synthetic, size=100
    ).dataloader()
    val_loader = SyntheticNeRFDataset(
        opt, device=device, type='val', H=opt.HW_vis, W=opt.HW_vis, size=5
    ).dataloader()
    vis_loader = SyntheticNeRFDataset(
        opt, device=device, type='test', H=opt.HW_vis, W=opt.HW_vis, size=100
    ).dataloader()
    test_loader = SyntheticNeRFDataset(
        opt, device=device, type='test', H=opt.HW_vis, W=opt.HW_vis, size=100
    ).dataloader()

    # Create loaders for real data
    real_train_loader = ImageOnlyNeRFDataset(
        opt, device=device, type='train', H=opt.HW_real, W=opt.HW_real, size=1
    ).dataloader()
    real_train_full_image_loader = ImageOnlyNeRFDataset(
        opt, device=device, type='train', H=opt.HW_real, W=opt.HW_real, size=1, force_test_mode=True
    ).dataloader()
    real_val_loader = ImageOnlyNeRFDataset(
        opt, device=device, type='val', H=opt.HW_real, W=opt.HW_real, size=8, load_image=False
    ).dataloader()
    real_test_loader = ImageOnlyNeRFDataset(
        opt, device=device, type='test', H=opt.HW_real, W=opt.HW_real, size=8, load_image=False
    ).dataloader()

    # Dataset length
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

    # Testing
    if opt.test:

        # Create loaders to sample camera views for testing and reconstruction
        test_loader = SyntheticNeRFDataset(opt, device=device, type='test', H=opt.HW_vis, W=opt.HW_vis, size=100).dataloader()

        # Create trainer
        trainer = Trainer(
            name='df', 
            opt=opt, 
            model=model, 
            guidance=None, 
            metrics=[],
            device=device, 
            workspace=opt.workspace, 
            fp16=opt.fp16, 
            use_checkpoint=opt.ckpt
        )

        # Setup
        trainer.prepare_for_reconstruction(real_train_loader, real_train_full_image_loader, real_val_loader, real_test_loader)

    else:

        # Stable diffusion guidance
        stable_diffusion_model = StableDiffusionModel.from_pretrained(opt.pretrained_model_name_or_path)
        if opt.learned_embeds_path is not None:  # add textual inversion tokens to model
            add_tokens_to_model_from_path(
                opt.learned_embeds_path, stable_diffusion_model.text_encoder, stable_diffusion_model.tokenizer
            )
        guidance = StableDiffusion(stable_diffusion_model=stable_diffusion_model, device=device)

        # Scheduler
        if opt.lr_warmup == 'vanilla':
            warm_up_with_cosine_lr = lambda iter: iter / opt.warm_iters if iter <= opt.warm_iters else max(0.5 * 
                (math.cos((iter - opt.warm_iters) /(opt.iters - opt.warm_iters) * math.pi) + 1), opt.min_lr / opt.lr)
            scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
        else:
            scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1)

        # Optimizer
        if opt.optim == 'adan':
            # Note: Adan usually requires a larger LR
            from optimizer import Adan
            optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
        else: # adamw
            optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        # Logging
        if opt.wandb:
            import wandb
            wandb.init(
                project='realfusion', name=Path(opt.workspace).name, job_type='train', 
                config=opt, save_code=True, sync_tensorboard=True,
            )

        # Create and prepare trainer
        trainer = Trainer(
            name='df',
            opt=opt,
            model=model,
            guidance=guidance,
            metrics=[],
            device=device,
            workspace=opt.workspace,
            optimizer=optimizer,
            ema_decay=opt.ema_decay,
            fp16=opt.fp16,
            lr_scheduler=scheduler,
            use_checkpoint=opt.ckpt,
            eval_interval=opt.eval_interval,
            scheduler_update_every_step=True
        )
        
        # Setup
        trainer.prepare_for_reconstruction(real_train_loader, real_train_full_image_loader, real_val_loader, real_test_loader)
        trainer.check_prompt()

        # Train
        trainer.train(train_loader, val_loader, vis_loader, max_epoch)

    # Test
    trainer.test(test_loader, name=opt.save_test_name)
    if opt.save_mesh:
        trainer.save_mesh(resolution=256)


if __name__ == '__main__':
    main()