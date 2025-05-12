import torch
from torch import tensor
import torch.nn.functional as F

from config import hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def positional_encode(positions: torch.Tensor, angles: torch.Tensor,
                       hyperparams=hyperparams):
    position_L = hyperparams["position_L"]
    angle_L = hyperparams["angle_L"]
    positions = positions * hyperparams["position_scale"]

    # Both positions.shape and angles.shape: [ray_num, samples_per_ray, 3]
    freqs = (torch.pi * positions.unsqueeze(-1) 
             * torch.pow(2, torch.arange(position_L, device=device))
             .view(1, 1, 1, -1))
    gamma_x = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=3)
    gamma_x = gamma_x.reshape(*gamma_x.shape[:2], -1) 

    freqs = (torch.pi * angles.unsqueeze(-1)
             * torch.pow(2, torch.arange(angle_L, device=device))
             .view(1, 1, 1, -1))
    gamma_d = torch.cat((torch.sin(freqs), torch.cos(freqs)), dim=3)
    gamma_d = gamma_d.reshape(*gamma_d.shape[:2], -1) 

    return gamma_x, gamma_d


def calculate_t_interval(camera_positions, ray_dirs, tmp_device=device):
    ray_dirs = ray_dirs + 1e-6 * (ray_dirs == 0).float() # to avoid zero division
    inv_dirs = 1 / ray_dirs
    t1 = (tensor([[-1, -1, -1]], device=tmp_device) - camera_positions) * inv_dirs
    t2 = (tensor([[1, 1, 1]], device=tmp_device) - camera_positions) * inv_dirs
    t_min = torch.maximum(torch.minimum(t1, t2),
                          tensor([[0, 0, 0]], device=tmp_device))
    t_max = torch.maximum(t1, t2)
    t_enters, _ = torch.max(t_min, dim=1)
    t_exits, _ = torch.min(t_max, dim=1)
    mask = t_enters < t_exits

    return t_enters[mask], t_exits[mask], mask


def coarse_sample(ray_orgs, ray_dirs, t_enters, t_exits,
                  hyperparams, train=True):
    batch_size = len(t_enters)
    sample_num = hyperparams["sample_coarse"]
    t_enters = t_enters.unsqueeze(1)
    t_exits = t_exits.unsqueeze(1)

    bins_ = torch.arange(sample_num, device=device).unsqueeze(0) # .expand(batch_size, sample_num)
    bins = bins_ * (t_exits - t_enters) / sample_num + t_enters
    if train:
        d_bins = torch.cat((bins[:, 1:] - bins[:, :-1],
                            t_exits - bins[:, -1].unsqueeze(1)), dim=1)
        ts = torch.rand(batch_size, sample_num, device=device) * d_bins + bins
    else:
        ts = bins
    dts = torch.cat(
        (ts[:, 1:] - ts[:, :-1], 
         torch.full((batch_size,), 1e10, device=device).unsqueeze(1)),
        dim=1
    )
    samples = (ray_orgs.unsqueeze(1)
               + ts.unsqueeze(2) * ray_dirs.unsqueeze(1))
    
    return samples, dts, ts


def fine_sample(ray_orgs, ray_dirs, ts, cdfs, hyperparams):
    batch_size = len(ts)
    sample_num = hyperparams["sample_fine"]

    u = torch.rand(batch_size, sample_num, device=device)

    idx = torch.searchsorted(cdfs, u, right=True)
    ts_mid = 0.5 * (ts[:, :-1] + ts[:, 1:])
    left = torch.clamp(idx - 1, min=0)
    right = torch.clamp(idx, max=cdfs.shape[-1] - 1)

    cdfs_left = torch.gather(cdfs, dim=1, index=left)
    cdfs_right = torch.gather(cdfs, dim=1, index=right)
    ts_left = torch.gather(ts_mid, dim=1, index=left)
    ts_right = torch.gather(ts_mid, dim=1, index=right)

    t = (u - cdfs_left) / torch.clamp(cdfs_right - cdfs_left, min=1e-6)
    ts_f = ts_left + t * (ts_right - ts_left)
    ts_f, _ = torch.cat((ts, ts_f), dim=-1).sort(dim=-1)
    dts_f = torch.cat(
        (ts_f[:, 1:] - ts_f[:, :-1], 
         torch.full((batch_size,), 1e10, device=device).unsqueeze(1)),
        dim=1
    )
    samples_f = (ray_orgs.unsqueeze(1)
                + ts_f.unsqueeze(2) * ray_dirs.unsqueeze(1))
    
    return samples_f, dts_f


def get_rays(fov, transform, hyperparams, tmp_device=device):
    focal_w_c = 2 * torch.tan(fov / 2)
    pixel_w_num = hyperparams["w_pixel_num"]
    pixel_w_c = focal_w_c / pixel_w_num
    pix_step = torch.linspace(
        (-focal_w_c + pixel_w_c) / 2,
        (focal_w_c - pixel_w_c) / 2, 
        pixel_w_num,
        device=tmp_device
    ) # [pixel_w_num,]
    x, y = torch.meshgrid(pix_step, -pix_step, indexing="xy")
    ray_to = (x.unsqueeze(-1) * tensor([1, 0, 0], device=tmp_device) 
              + y.unsqueeze(-1) * tensor([0, 1, 0], device=tmp_device)
              + tensor([0, 0, -1], device=tmp_device)) # [pixel_w_num, pixel_w_num, 3]
    ray_dirs = F.normalize(ray_to, dim=-1)
    ray_dirs_w = (ray_dirs @ transform[:3, :3].T).reshape(-1, 3) # [pixel_w_num * pixel_w_num, 3]
    ray_orgs_w = transform[:3, 3].unsqueeze(0).expand(ray_dirs_w.shape[0], 3)
    
    return ray_orgs_w, ray_dirs_w # [pixel_w_num * pixel_w_num, 3]


def get_rays_debug(fov, transform, hyperparams):
    focal_w_c = 2 * torch.tan(fov / 2)
    pixel_w_num = hyperparams["w_pixel_num"]
    pixel_w_c = focal_w_c / pixel_w_num
    pix_step = torch.linspace(
        (-focal_w_c + pixel_w_c) / 2,
        (focal_w_c - pixel_w_c) / 2, 
        pixel_w_num,
        device=device
    ) # [pixel_w_num,]
    x, y = torch.meshgrid(pix_step, -pix_step, indexing="xy")
    ray_to = (x.unsqueeze(-1) * tensor([1, 0, 0], device=device) 
              + y.unsqueeze(-1) * tensor([0, 1, 0], device=device)
              + tensor([0, 0, -1], device=device)) # [pixel_w_num, pixel_w_num, 3]
    ray_dirs = F.normalize(ray_to, dim=-1)
    ray_dirs_w = (ray_dirs @ transform[:3, :3].T).reshape(-1, 3) # [pixel_w_num * pixel_w_num, 3]
    ray_orgs_w = transform[:3, 3:].T.expand(ray_dirs_w.shape[0], 3)
    
    return ray_orgs_w, ray_dirs_w, ray_to.reshape(-1, 3) # [pixel_w_num * pixel_w_num, 3]


def get_focal_plane(fov, transform):
    focal_w_c = 2 * torch.tan(fov / 2)
    to_focal = tensor([
        [0, 0, -1],
        [-focal_w_c, focal_w_c, -1],
        [focal_w_c, focal_w_c, -1],
        [-focal_w_c, -focal_w_c, -1],
        [focal_w_c, -focal_w_c, -1]
    ])
    to_focal_w = (to_focal @ transform[:3, :3].T).reshape(-1, 3)
    ray_orgs_w = transform[:3, 3:].T.expand(to_focal_w.shape[0], 3)

    return ray_orgs_w, to_focal_w

