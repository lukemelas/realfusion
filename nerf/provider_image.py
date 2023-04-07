import random
import numpy as np

import trimesh
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TVF
from PIL import Image

from .utils import get_rays, safe_normalize
from .options import Options


DIR_COLORS = np.array([
    [255, 0, 0, 255],  # front
    [0, 255, 0, 255],  # side
    [0, 0, 255, 255],  # back
    [255, 255, 0, 255],  # side
    [255, 0, 255, 255],  # overhead
    [0, 255, 255, 255],  # bottom
], dtype=np.uint8)


def visualize_poses(poses, dirs, size=0.1):
    # poses: [B, 4, 4], dirs: [B]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose, dir in zip(poses, dirs):
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)

        # different color for different dirs
        segs.colors = DIR_COLORS[[dir]].repeat(len(segs.entities), 0)

        objects.append(segs)

    trimesh.Scene(objects).show()


def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def rand_poses(
        size, device, radius_range=[1, 1.5],
        theta_range=[0, 120],
        phi_range=[0, 360],
        return_dirs=False, angle_overhead=30, angle_front=60, jitter=False, uniform_sphere_rate=0.5):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]

    if random.random() < uniform_sphere_rate:
        unit_centers = F.normalize(
            torch.stack([
                (torch.rand(size, device=device) - 0.5) * 2.0,
                torch.rand(size, device=device),
                (torch.rand(size, device=device) - 0.5) * 2.0,
            ], dim=-1), p=2, dim=1
        )
        thetas = torch.acos(unit_centers[:, 1])
        phis = torch.atan2(unit_centers[:, 0], unit_centers[:, 2])
        phis[phis < 0] += 2 * np.pi
        centers = unit_centers * radius.unsqueeze(-1)
    else:
        thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
        phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

        centers = torch.stack([
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ], dim=-1)  # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))

    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None

    return poses, dirs


class NeRFDataset:
    def __init__(self, opt: Options, device, type='train', H=256, W=256, size=100, radius=None, load_image=True, force_test_mode=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test

        self.H = H
        self.W = W
        self.radius_range = opt.radius_range
        self.fovy_range = opt.fovy_range
        self.size = size

        self.training = False if force_test_mode else (self.type in ['train', 'all'])

        self.cx = self.H / 2
        self.cy = self.W / 2

        radius_rot = self.opt.radius_rot if self.opt.radius_rot else (self.radius_range[1] * 1.2)
        self.circle_radius = radius if radius else radius_rot

        self.num_rays = self.opt.num_rays if self.training else -1
            
        ### Real images for reconstruction ###
        if load_image:

            # Load RGBA image
            self.image = TVF.to_tensor(Image.open(self.opt.image_path)).unsqueeze(0).to(self.device)  # (1, 3, H, W)
            self.image: Tensor = F.interpolate(self.image, size=(H, W), mode='bilinear')
            self.image = self.image.permute(0, 2, 3, 1)  # (1, H, W, 4), with 4 for RGBA
            self.C = self.image.shape[-1]
        
        else:
            self.image = None

    def collate(self, index):

        B = len(index)
        error_map = None

        # circle pose
        phi = (index[0] / self.size) * 360
        poses, dirs = circle_poses(
            self.device, radius=self.circle_radius, phi=phi, theta=self.opt.pose_angle,
            return_dirs=self.opt.dir_text, angle_overhead=self.opt.angle_overhead, angle_front=self.opt.angle_front
        )

        # fixed focal
        fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
        focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
        intrinsics = np.array([focal, focal, self.cx, self.cy])

        # sample rays
        rays = get_rays(poses, intrinsics, self.H, self.W, self.num_rays, error_map)

        data = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            # 'dir': dirs,  # no dirs for image (at the moment)
        }

        if self.image is not None:
            data['image_full_size'] = image = self.image
            if self.training:
                image = torch.gather(image.view(B, -1, self.C), 1, torch.stack(self.C * [rays['inds']], -1))  # [B, N, C], C=4
            data['images'] = image

        return data

    def dataloader(self, shuffle=None):
        shuffle = self.training if shuffle is None else shuffle
        loader = DataLoader(list(range(self.size)), batch_size=1,
                            collate_fn=self.collate, shuffle=shuffle, num_workers=0)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader
