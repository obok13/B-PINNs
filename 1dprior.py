#%%

import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import util

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# hyperparameters

hamiltorch.set_random_seed(123)
prior_std = 1
like_std = 0.1
step_size = 0.1
burn = 1000
num_samples = 5000
L = 500
layer_sizes = [1,16,16,1]
activation = torch.tanh
pde = False
pinns = False
epochs = 10000
tau_priors = [1,1,1,1,1,1]
tau_likes = 1/like_std**2

lb = -1
ub = 1
N_val = 100

# data

data_val = {}
data_val['x'] = torch.linspace(lb,ub,N_val).view(-1,1)
data_val['x'] = data_val['x'].to(device)

# model

class Net(nn.Module):

    def __init__(self, layer_sizes, activation=torch.tanh):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        # self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        # x = self.activation(x)
        # x = self.l4(x)
        return x

net_u = Net(layer_sizes, activation).to(device)
nets = [net_u]

def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    x = data['x'].to(device)
    pred = fmodel[0](x, params=params_unflattened[0])
    ll = 0
    output = [pred]

    if torch.cuda.is_available():
        del x
        torch.cuda.empty_cache()

    return ll, output

# sampling

params_hmc = util.sample_model_bpinns(nets, data_val, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L, step_size=step_size, burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde = pde, pinns=pinns, epochs=epochs)

pred_list, log_prob_list = util.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss, tau_priors=tau_priors, tau_likes=tau_likes, pde = pde)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

pred_list_u = pred_list[0].cpu().numpy()

# plot

cov = np.cov(pred_list_u.squeeze().T)
extent = [lb,ub,lb,ub]
plt.figure(figsize=(5,5))
plt.imshow(cov, extent=extent)
plt.colorbar()
# %%
