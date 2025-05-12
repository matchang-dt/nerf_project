import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from config import hyperparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeRF(nn.Module):
    def __init__(self, hyperparams=hyperparams):
        super(NeRF, self).__init__()

        hidden_d = hyperparams['hidden_dimension']
        mlp_num = hyperparams['mlp_num']
        res_at = hyperparams['res_at']
        if res_at >= mlp_num or res_at < 0:
            raise ValueError('inappropriate res_at')
        self.position_L = hyperparams['position_L']
        self.angle_L = hyperparams['angle_L']
        self.res_at = res_at
        
        if res_at > 0:
            layers1 = [nn.Linear(self.position_L * 6, hidden_d), nn.ReLU()]
            for _ in range(res_at - 1):
                layers1.append(nn.Linear(hidden_d, hidden_d))
                layers1.append(nn.ReLU())
            self.encoder1 = nn.Sequential(*layers1)

            layers2 = [
                nn.Linear(hidden_d + self.position_L * 6, hidden_d),
                nn.ReLU()
            ]
            for _ in range(mlp_num - res_at - 1):
                layers2.append(nn.Linear(hidden_d, hidden_d))
                layers2.append(nn.ReLU())
            layers2.append(nn.Linear(hidden_d, hidden_d + 1))
            self.encoder2 = nn.Sequential(*layers2)
        else:
            layers1 = [nn.Linear(self.position_L * 6, hidden_d), nn.ReLU()]
            for _ in range(mlp_num - 1):
                layers1.append(nn.Linear(hidden_d, hidden_d))
                layers1.append(nn.ReLU())
            self.encoder1 = nn.Sequential(*layers1)
            self.encoder2 = nn.Linear(hidden_d, hidden_d + 1)

        self.radiance_net = nn.Sequential(
            nn.Linear(hidden_d + self.angle_L * 6, hidden_d // 2),
            nn.ReLU(),
            nn.Linear(hidden_d // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, gamma_x, gamma_d):
        hidden = self.encoder1(gamma_x)
        if self.res_at > 0:
            hidden = torch.cat((hidden, gamma_x), dim=-1)
        encoded = self.encoder2(hidden)
        pos_feat = encoded[..., :-1]
        sigma = torch.relu(encoded[..., -1])
        radiance = self.radiance_net(torch.cat((pos_feat, gamma_d), dim=-1))
        return radiance, sigma

if __name__ == "__main__":
    coarse_net = NeRF(hyperparams)
    print(coarse_net)
    print(coarse_net.encoder1[0].weight)
    for m in coarse_net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('sigmoid'))
    print(coarse_net.encoder1[0].weight)


