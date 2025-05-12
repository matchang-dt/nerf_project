import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from nerf import NeRF
from config import hyperparams
from rendering import coarse_rendering, fine_rendering
from util import get_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(base_dir, input_json, hyperparams=hyperparams, num_samples=None):
    with open(input_json, 'r') as f:
        data = json.load(f)

    # fov = tensor(data["camera_angle_x"], device=device)
    fov = tensor(data["camera_angle_x"], device=device)
    w_pix_num = hyperparams["w_pixel_num"]

    base_name = os.path.basename(input_json)
    save_path = os.path.join(base_dir,
                             base_name.split(".")[0] + "_pixels.pth")

    pixels = [] # pixel valuew of the images (label)
    ray_orgs = []
    ray_dirs = []
    t_near = []
    t_far = []

    count = 0
    for frame in tqdm(data["frames"], desc="Data processing"):
        transform = torch.tensor(frame["transform_matrix"], device=device)
        ray_orgs_img, ray_dirs_img = get_rays(fov, transform,
                                            hyperparams, tmp_device=device)
        ray_orgs.append(ray_orgs_img)
        ray_dirs.append(ray_dirs_img)

        if not os.path.exists(save_path):
            fpath = os.path.join(base_dir, frame["file_path"][2:] + ".png")
            with Image.open(fpath) as img:
                img_np = np.array(img, dtype=np.float32) / 255
                img = (img_np[..., :3] * img_np[..., 3:] 
                    + (1. - img_np[..., 3:]) 
                    * np.array([1., 1., 1.], dtype=np.float32))
                img = img.clip(0., 1.)
                pixels.append(img.reshape(-1, 3))
        count += 1
        if num_samples and count >= num_samples:
            break

    ray_orgs = torch.cat(ray_orgs, dim=0)
    ray_dirs = torch.cat(ray_dirs, dim=0)
    if not os.path.exists(save_path):
        pixels = np.concatenate(pixels, axis=0)
        pixels = torch.from_numpy(pixels)
        torch.save(pixels, save_path)
        print(f"Saved pixel data to {save_path}")
    else:
        print(f"Pixel data already exists at {save_path}")
        print("Loading pixel data...")
        pixels = torch.load(save_path)
    pixels = pixels.to(device)

    t_near = torch.full_like(ray_orgs[..., 0], hyperparams["t_near"],
                             device=device)
    t_far = torch.full_like(ray_orgs[..., 0], hyperparams["t_far"],
                            device=device)

    return ray_orgs, ray_dirs, t_near, t_far, pixels


def validate(coarse_net, fine_net, val_loader,
             hyperparams=hyperparams, trans_func=torch.exp):
    with torch.no_grad():
        val_loss = 0
        criterion = nn.MSELoss()
        for data, pix_b in val_loader:
            orgs_b = data["orgs"]
            dirs_b = data["dirs"]
            t_n_b = data["t_n"]
            t_f_b = data["t_f"]

            _, ts_c, dts_c, cdfs = coarse_rendering(
                orgs_b, dirs_b, t_n_b, t_f_b,
                coarse_net, trans_func
            )
            pix_pred = fine_rendering(
                orgs_b, dirs_b, ts_c, cdfs, fine_net, trans_func
            )
            batch_loss = criterion(pix_pred, pix_b)
            val_loss += batch_loss.item()
        val_loss /= len(val_loader)
        psnr = -10 * math.log10(val_loss)
    return val_loss, psnr

    
class NeRFDataset(Dataset):
    def __init__(self, ray_orgs, ray_dirs, t_near, t_far, pixels):
        self.orgs = ray_orgs
        self.dirs = ray_dirs
        self.t_n = t_near
        self.t_f = t_far
        self.pixels = pixels

    def __len__(self):
        return len(self.orgs)

    def __getitem__(self, idx):
        return {
            "orgs": self.orgs[idx],
            "dirs": self.dirs[idx],
            "t_n": self.t_n[idx],
            "t_f": self.t_f[idx]
        }, self.pixels[idx]

if __name__ == "__main__":
    base_dir = "data/nerf_synthetic/lego"
    input_json = os.path.join(base_dir, "transforms_train_100.json")
    val_json = os.path.join(base_dir, "transforms_val_100.json")
    ray_orgs, ray_dirs, t_near, t_far, pixels = preprocess(
        base_dir, input_json, hyperparams=hyperparams
    )
    print(pixels.shape)
    print(ray_orgs.shape)
    print(ray_dirs.shape)
    print(t_near.shape)
    print(t_far.shape)

