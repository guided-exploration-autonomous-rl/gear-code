import math

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
import numpy as np
# from mbrl.third_party.pytorch_sac import utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.optim as optim
from torch.distributions import Categorical

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class AutoRegressiveModel(nn.Module):
    # for trajectory: action dimension is the horizon and number of buckets is the dimension of action space  
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, num_buckets=10, ac_low=-1, ac_high=1):        
        super().__init__()
        self.eps = 1e-8
        self.trunks = nn.ModuleList([mlp(obs_dim, hidden_dim, num_buckets, hidden_depth)] \
                        + [mlp(obs_dim + j + 1, hidden_dim, num_buckets, hidden_depth) for j in range(action_dim - 1)])
        self.num_dims = action_dim
        self.ac_low = torch.Tensor(ac_low).to(device)
        self.ac_high = torch.Tensor(ac_high).to(device)
        self.num_buckets = num_buckets
        self.bucket_size = torch.Tensor((ac_high - ac_low) / num_buckets).to(device)
         
    def discretize(self, ac):
        bucket_idx = (ac - self.ac_low) // (self.bucket_size + self.eps)
        return torch.clip(bucket_idx, 0, self.num_buckets - 1)
    
    def undiscretize(self, bucket_idx, dimension):
        return_val = bucket_idx[:, None]*self.bucket_size + self.ac_low + self.bucket_size*0.5
        return return_val[:, dimension]
        
    def forward(self, obs):
        vals = []
        for j in range(self.num_dims):
            if j == 0:
                in_val = obs
            else:
                in_val = torch.cat([in_val, ac_prev[:, None]], dim=-1)
            logit = self.trunks[j](in_val)
            dist_samples = Categorical(logits=logit).sample()
            vals.append(self.undiscretize(dist_samples, j)[:, None])
            ac_prev = dist_samples
        
        vals = torch.cat(vals, dim=-1)
        return vals
    
    def log_prob(self, obs, act):
        log_prob = 0.
        ac_discretized = self.discretize(act)
        for j in range(self.num_dims):
            if j == 0:
                in_val = obs
            else:
                in_val = torch.cat([in_val, ac_prev[:, None]], dim=-1)
            logit = self.trunks[j](in_val)
            dist = Categorical(logits=logit)
            lp = dist.log_prob(ac_discretized[:, j])
            log_prob += lp
            ac_prev = ac_discretized[:,j]
        return log_prob

    def train(self, data, num_epochs=500, batch_size=64, plotfreq=1):
        optimizer = optim.Adam(self.parameters())
        num_data = len(data)
        losses = []
        idxs = np.array(range(len(data['obs'])))
        num_batches = len(idxs) // batch_size
        # Train the model with regular SGD
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            np.random.shuffle(idxs)
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()

                curr_idx1 = idxs[batch_size*i: batch_size*i + batch_size] 
                obs_curr = torch.from_numpy(data['obs'][curr_idx1]).to(device).float()
                pol_reps_curr = torch.from_numpy(data['policy'][curr_idx1]).to(device).float()
                in_val = torch.cat([obs_curr, pol_reps_curr], dim=-1)
                out_val = torch.from_numpy(data['next_obs'][curr_idx1]).to(device).float()
                loss = -torch.mean(self.log_prob(in_val, out_val))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.cpu().detach().numpy()
                if i % plotfreq == 0 and i > 0:    # print every 2000 mini-batches
                    print('[%d, %5d] full loss: %.6f' %
                        (epoch + 1, i + 1, running_loss / 100.))
                    losses.append(running_loss/100.)
                    running_loss = 0.0

        print('Finished Training')
        return losses