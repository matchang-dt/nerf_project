from datetime import datetime
from itertools import chain
from time import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from nerf import NeRF
from config import hyperparams
from rendering import coarse_rendering, fine_rendering
from train_util import preprocess, NeRFDataset, validate
from util import positional_encode, coarse_sample, fine_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trans_func = torch.exp

def train(base_dir, train_json, val_json, save_folder, trans_func = torch.exp,
          resumption:dict = None):
    start = time()
    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    model_folder = os.path.join(save_folder, now_str + "/")
    os.makedirs(os.path.dirname(model_folder), exist_ok=True)
    save_hyperparams = f"hparams_{now_str}.pth"
    torch.save(hyperparams, os.path.join(model_folder, save_hyperparams))
    csv_path = os.path.join(model_folder, f"log_{now_str}.csv")
    with open(csv_path,"a", encoding="utf-8") as f:
        f.write("epoch,step,coarse,fine\n")
    coarse_file = f"coarse_{now_str}.pth" # to save param
    fine_file = f"fine_{now_str}.pth" # to save param

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
    epochs = hyperparams["epochs"]
    gamma_lr = ((hyperparams["decay_to"] / hyperparams["lr"]) 
            ** (1 / epochs))
    params = list(coarse_net.parameters()) + list(fine_net.parameters())
    lr = hyperparams["lr"]
    if resumption:
        epochs = resumption["epochs"] - resumption["done"]
        lr = hyperparams["lr"] * pow(gamma_lr, resumption["done"])
        id = resumption["id"]
        coarse_path = f"models/{id}/coarse_{id}.pth"
        fine_path = f"models/{id}/fine_{id}.pth"
        if torch.cuda.is_available():
            state_dict_c = torch.load(coarse_path)
            state_dict_f = torch.load(fine_path)
        else:
            state_dict_c = torch.load(coarse_path, map_location=torch.device('cpu'))
            state_dict_f = torch.load(fine_path, map_location=torch.device('cpu'))
        coarse_net.load_state_dict(state_dict_c)
        fine_net.load_state_dict(state_dict_f)

    optimizer = optim.AdamW(params, lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_lr)
    criterion = nn.MSELoss()

    ray_orgs, ray_dirs, t_near, t_far, pixels = preprocess(
        base_dir, train_json, hyperparams=hyperparams
    )
    train_loader = DataLoader(
        NeRFDataset(ray_orgs, ray_dirs, t_near, t_far, pixels),
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=True
    )

    val_sample = 3
    ray_orgs, ray_dirs, t_near, t_far, pixels = preprocess(
        base_dir, val_json, hyperparams=hyperparams, num_samples=val_sample
    )
    val_loader = DataLoader(
        NeRFDataset(ray_orgs, ray_dirs, t_near, t_far, pixels),
        batch_size=hyperparams["w_pixel_num"] * 4,
        shuffle=False,
    )

    log_counter = 0
    step = 0
    end_flg = False
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1} / {epochs}")
        for data, pix_b in pbar:
            orgs_b = data["orgs"]
            dirs_b = data["dirs"]
            t_n_b = data["t_n"]
            t_f_b = data["t_f"]
            optimizer.zero_grad()

            pix_pred, ts_c, dts_c, cdfs = coarse_rendering(orgs_b, dirs_b,
                                                            t_n_b, t_f_b,
                                                            coarse_net,
                                                            trans_func)
            loss1 = criterion(pix_pred, pix_b)
            pix_pred = fine_rendering(orgs_b, dirs_b, ts_c, cdfs,
                                        fine_net, trans_func)
            loss2 = criterion(pix_pred, pix_b)
            loss = loss1 + loss2           
            loss.backward()

            if not torch.isfinite(loss):
                if torch.isnan(loss):
                    print("NaN detected, stopping training.")
                    end_flg = True
                    break
                else:
                    print("Wrong values detected, stopping training.")
                    end_flg = True
                    break
            optimizer.step()
            log_counter += 1
            pbar.set_postfix(
                loss=loss.item(), 
                coarse_loss=loss1.item(),
                fine_loss=loss2.item()
            )
            step += 1
            if log_counter == 100:
                with open(csv_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch+1},{step},{loss1:.3g},{loss2:.3g}\n")
                torch.save(coarse_net.state_dict(),
                            os.path.join(model_folder, coarse_file))
                torch.save(fine_net.state_dict(),
                            os.path.join(model_folder, fine_file))
                log_counter = 0
        if end_flg:
            break
        pbar.close()
        val_loss, psnr = validate(coarse_net, fine_net, val_loader,
                                  hyperparams=hyperparams,
                                  trans_func=trans_func)
        print(f"Validation loss: {val_loss:.3g}, PSNR: {psnr:.3g}")
        torch.save(coarse_net.state_dict(),
                    os.path.join(model_folder, coarse_file))
        torch.save(fine_net.state_dict(),
                    os.path.join(model_folder, fine_file))
        scheduler.step()

    print("Finished!")
    print(f"Elapsed: {(time() - start):.1f} s")

if __name__ == "__main__":    
    base_dir = "data/nerf_synthetic/lego"
    train_json = os.path.join(base_dir, "transforms_train_100.json")
    val_json = os.path.join(base_dir, "transforms_val_100.json")
    save_folder = "models/"
    train(base_dir, train_json, val_json, save_folder, trans_func = torch.exp)