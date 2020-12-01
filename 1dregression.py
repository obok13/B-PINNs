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
step_size = 0.001
burn = 500
num_samples = 1500
L = 100
layer_sizes = [1,16,16,1]
activation = torch.tanh
model_loss = '1dregression'
pde = False
pinns = False
epochs = 10000
tau_priors = [1,1,1,1,1,1]
tau_likes = [1/like_std**2]

lb = -1
ub = 1
N_tr = 32
N_val = 100

# data

def u(x):
    return torch.sin(6*x)**3
data = {}
x1 = torch.linspace(-0.8,-0.2,int(N_tr/2))
x2 = torch.linspace(0.2,0.8,int(N_tr/2))
data['x'] = torch.cat((x1,x2),0).view(-1,1)
data['y'] = u(data['x']) + torch.randn_like(data['x'])*like_std

data_val = {}
data_val['x'] = torch.linspace(lb,ub,N_val).view(-1,1)
data_val['y'] = u(data_val['x'])

data['x'] = data['x'].to(device)
data['y'] = data['y'].to(device)

data_val['x'] = data_val['x'].to(device)
data_val['y'] = data_val['y'].to(device)

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

# sampling

params_hmc = util.sample_model_bpinns(nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L, step_size=step_size, burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde = pde, pinns=pinns, epochs=epochs)

pred_list, log_prob_list = util.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss, tau_priors=tau_priors, tau_likes=tau_likes, pde = pde)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

pred_list = pred_list[0].cpu().numpy()

# plot

x_val = data_val['x'].cpu().numpy()
u_val = data_val['y'].cpu().numpy()
x_u = data['x'].cpu().numpy()
y_u = data['y'].cpu().numpy()

plt.figure(figsize=(7,5))
plt.plot(x_val,u_val,'r-',label='Exact')
# plt.plot(x_val,pred_list.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val,pred_list.mean(0).squeeze().T, 'b-',alpha=0.9,label ='Mean')
plt.fill_between(x_val.reshape(-1), pred_list.mean(0).squeeze().T - 2*pred_list.std(0).squeeze().T, pred_list.mean(0).squeeze().T + 2*pred_list.std(0).squeeze().T, facecolor='b', alpha=0.2, label = '2 std')
plt.plot(x_u,y_u,'kx',markersize=5, label='Training data')
plt.xlim([lb,ub])
plt.legend(fontsize=10)
plt.show()
# %%
