import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import tensor
from tqdm import tqdm

from nerf import NeRF
from config import hyperparams
from util import positional_encode, coarse_sample, fine_sample, get_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coarse_rendering(rays_org, rays_dir, ts_near, ts_far,
                     coarse_net, trans_func, train=True):
    samples, dts, ts = coarse_sample(rays_org, rays_dir, ts_near,
                                            ts_far, hyperparams, train)
    dir_b_ = (rays_dir.unsqueeze(1).expand(rays_org.shape[0], 
              hyperparams["sample_coarse"], 3))
    gamma_x, gamma_d = positional_encode(samples, dir_b_, hyperparams)
    radiances, sigmas = coarse_net.forward(gamma_x, gamma_d)
    sigmas = torch.clamp(sigmas, min=1e-6, max=100)
    section_trans = trans_func(-sigmas * dts)
    section_trans = torch.clamp(section_trans, min=1e-6) 
    transmittances = torch.cumprod(section_trans, dim=-1) 
    transmittances = torch.cat(
        (torch.ones(*transmittances.shape[:-1], 1, device=device),
         transmittances[...,:-1]), dim=-1
    )
    weights = transmittances * (1 - section_trans)
    cum_weights = torch.cumsum(weights[:, :-1], dim=1)
    cum_weights[:, -1] = (cum_weights[:, -1] 
                          + (cum_weights[:, -1] == 0).float() * 1e-6)
    cdfs = cum_weights / cum_weights[:, -1:]
    pixels_color = torch.sum(weights.unsqueeze(-1) * radiances, dim=1)
    
    return pixels_color, ts, dts, cdfs


def fine_rendering(rays_org, rays_dir, ts_c, cdfs, fine_net, trans_func):
    samples_f, dts_f = fine_sample(rays_org, rays_dir, ts_c,
                                    cdfs, hyperparams)
    dir_b_ = (rays_dir.unsqueeze(1).expand(*samples_f.shape))
    dir_b_ = dir_b_[:, 0, :].unsqueeze(1).expand(*samples_f.shape)
    gamma_x, gamma_d = positional_encode(samples_f, dir_b_, hyperparams)
    radiances, sigmas = fine_net.forward(gamma_x, gamma_d)
    sigmas = torch.clamp(sigmas, min=1e-6, max=100)
    section_trans = trans_func(-sigmas * dts_f)
    section_trans = torch.clamp(section_trans, min=1e-6) 
    transmittances = torch.cumprod(section_trans, dim=-1) 
    transmittances = torch.cat(
        (torch.ones(*transmittances.shape[:-1], 1, device=device),
         transmittances[...,:-1]), dim=-1
    )
    weights = transmittances * (1 - section_trans)
    pixels_color = torch.sum(weights.unsqueeze(-1) * radiances, dim=1)
    
    return pixels_color


def set_axes_equal(ax, width=4): # for debug
    '''Align x, y, z scales'''
    ax.set_xlim3d([-width, width])
    ax.set_ylim3d([-width, width])
    ax.set_zlim3d([-width, width])


def render_with_json(input_json, data_idx, hparam_path=None,
                     coarse_path=None, fine_path=None, trans_func = torch.exp,
                     coarse_use=False):
    if hparam_path:
        hyperparams = torch.load(hparam_path)
    else:
        from config import hyperparams
    coarse_net = NeRF(hyperparams).to(device)
    fine_net = NeRF(hyperparams).to(device)
    for m in coarse_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 
                                    gain=nn.init.calculate_gain('sigmoid'))
    for m in fine_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 
                                    gain=nn.init.calculate_gain('sigmoid'))
    if coarse_path:
        state_dict = torch.load(coarse_path, map_location=torch.device('cpu'))
        coarse_net.load_state_dict(state_dict)
    if fine_path:
        state_dict = torch.load(fine_path, map_location=torch.device('cpu'))
        fine_net.load_state_dict(state_dict)

    with open(input_json, 'r') as f:
        data = json.load(f)

    fov = tensor(data["camera_angle_x"], device=device)    
    transform = tensor(
        data["frames"][data_idx]["transform_matrix"], device=device
    )

    ray_orgs, ray_dirs = get_rays(fov, transform, hyperparams) # [pix_num_w ** 2, 3]
    t_enters = torch.full_like(ray_orgs[..., 0], hyperparams["t_near"],
                               device=device)
    t_exits = torch.full_like(ray_orgs[..., 0], hyperparams["t_far"],
                              device=device)
    pix_num_w = hyperparams["w_pixel_num"]
    img = torch.zeros((pix_num_w, pix_num_w, 3), device=device)

    for i in tqdm(range(pix_num_w), desc="rendering"):
        ray_orgs_r = ray_orgs[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r, 3]
        ray_dirs_r = ray_dirs[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r, 3]
        t_enters_r = t_enters[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r,]
        t_exits_r = t_exits[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r,]

        with torch.no_grad():
            pixels_color, ts_c, dts_c, cdfs = coarse_rendering(
                rays_org=ray_orgs_r,
                rays_dir=ray_dirs_r,
                ts_near=t_enters_r,
                ts_far=t_exits_r,
                coarse_net=coarse_net,
                trans_func=trans_func,
                train=False
            )
            if not coarse_use:
                pixels_color = fine_rendering(ray_orgs_r, ray_dirs_r, ts_c,
                                              cdfs, fine_net, trans_func)
            img[i, :] = pixels_color

    img_numpy = img.to("cpu").numpy()
    plt.imshow(img_numpy)
    plt.show()


def render_from_view_point(fov, x, y, z, hparam_path=None, coarse_path=None,
                           fine_path=None, trans_func = torch.exp,
                           coarse_use=False):
    if hparam_path:
        hyperparams = torch.load(hparam_path)
    else:
        from config import hyperparams
    coarse_net = NeRF(hyperparams).to(device)
    fine_net = NeRF(hyperparams).to(device)
    for m in coarse_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 
                                    gain=nn.init.calculate_gain('sigmoid'))
    for m in fine_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, 
                                    gain=nn.init.calculate_gain('sigmoid'))
    if coarse_path:
        state_dict = torch.load(coarse_path, map_location=torch.device('cpu'))
        coarse_net.load_state_dict(state_dict)
    if fine_path:
        state_dict = torch.load(fine_path, map_location=torch.device('cpu'))
        fine_net.load_state_dict(state_dict)
    
    fov = tensor(fov, device=device)
    p_column = tensor([x, y, z, 1.], device=device)
    forward = -F.normalize(p_column[:3])
    f_column = torch.cat(-forward, tensor(0., device=device))

    if y < 0.999:
        u0 = tensor([0., 1., 0.], device=device)
    else:
        u0 = tensor([1., 0., 0.], device=device)
    r0 = torch.cross(forward, u0)
    up = torch.cross(r0, forward)
    u_column = torch.cat(up, tensor(0., device=device))
    right = torch.cross(forward, up)
    r_column = torch.cat(right, tensor(0., device=device))
    transform = torch.stack((r_column, u_column, f_column, p_column), dim=1)

    ray_orgs, ray_dirs = get_rays(fov, transform, hyperparams) # [pix_num_w ** 2, 3]
    t_enters = torch.full_like(ray_orgs[..., 0], hyperparams["t_near"],
                               device=device)
    t_exits = torch.full_like(ray_orgs[..., 0], hyperparams["t_far"],
                              device=device)
    pix_num_w = hyperparams["w_pixel_num"]
    img = torch.zeros((pix_num_w, pix_num_w, 3), device=device)

    for i in tqdm(range(pix_num_w), desc="rendering"):
        ray_orgs_r = ray_orgs[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r, 3]
        ray_dirs_r = ray_dirs[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r, 3]
        t_enters_r = t_enters[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r,]
        t_exits_r = t_exits[i * pix_num_w:(i + 1) * pix_num_w] # [eff_num_r,]

        with torch.no_grad():
            pixels_color, ts_c, dts_c, cdfs = coarse_rendering(
                rays_org=ray_orgs_r,
                rays_dir=ray_dirs_r,
                ts_near=t_enters_r,
                ts_far=t_exits_r,
                coarse_net=coarse_net,
                trans_func=trans_func,
                train=False
            )
            if not coarse_use:
                pixels_color = fine_rendering(ray_orgs_r, ray_dirs_r, ts_c,
                                              cdfs, fine_net, trans_func)
            img[i, :] = pixels_color

    img_numpy = img.to("cpu").numpy()
    plt.imshow(img_numpy)
    plt.show()


if __name__ == "__main__":
    folder = "250512024237"
    if folder:
        hparam_path = f"models/{folder}/hparams_{folder}.pth"
        coarse_path = f"models/{folder}/coarse_{folder}.pth"
        fine_path = f"models/{folder}/fine_{folder}.pth"
    else:
        hparam_path = "models/hparams_250418102603.pth"
        coarse_path = "models/coarse_250418102603.pth"
        fine_path = "models/fine_250418102603.pth"
    
    base_dir = "data/nerf_synthetic/lego"
    # input_json = os.path.join(base_dir, "transforms_train_100.json")
    input_json = os.path.join(base_dir, "transforms_val_100.json")
    # input_json = os.path.join(base_dir, "transforms_test_100.json")
    data_idx = 1
    # render_with_json(input_json, data_idx, hparam_path, coarse_path, fine_path)
    render_with_json(input_json, data_idx, hparam_path, coarse_path, fine_path, 
                     coarse_use=True)

