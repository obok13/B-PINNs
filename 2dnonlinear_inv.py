#%%

import torch
import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import util

# device

print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# hyperparameters

hamiltorch.set_random_seed(123)
prior_std = 1
like_std = 0.1
step_size = 0.001
burn = 200
num_samples = 400
L = 100
layer_sizes = [2,16,16,1]
activation = torch.tanh
model_loss = '2dnonlinear_inv'
pde = True
pinns = False
epochs = 10000
tau_priors = 1/prior_std**2
tau_likes = 1/like_std**2

lb = -1
ub = 1
N_tr_u = 100
N_tr_f = 100
N_tr_b = 25
N_val = 100

# data

exact_single = np.array([1.])

def u(x):
    return torch.sin(np.pi*x[:,0:1]) * torch.sin(np.pi*x[:,1:2])
def f(x):
    return 0.01 * -2*np.pi**2 * u(x) + exact_single[0]*u(x)**2

data = {}
xb = torch.linspace(lb,ub,N_tr_b)
xb = torch.cartesian_prod(xb,xb)
xb = xb[torch.sum((xb==1) + (xb==-1),1).bool(),:]
data['x_u'] = torch.cat((xb,(ub-lb)*torch.rand(N_tr_u,2)+lb),0)
data['y_u'] = u(data['x_u']) + torch.randn_like(u(data['x_u']))*like_std
data['x_f'] = (ub-lb)*torch.rand(N_tr_f,2)+lb
data['y_f'] = f(data['x_f']) + torch.randn_like(f(data['x_f']))*like_std

data_val = {}
xu = torch.linspace(lb,ub,N_val)
data_val['x_u'] = torch.cartesian_prod(xu,xu)
data_val['y_u'] = u(data_val['x_u'])
data_val['x_f'] = torch.cartesian_prod(xu,xu)
data_val['y_f'] = f(data_val['x_f'])

data['x_u'] = data['x_u'].to(device)
data['y_u'] = data['y_u'].to(device)
data['x_f'] = data['x_f'].to(device)
data['y_f'] = data['y_f'].to(device)

data_val['x_u'] = data_val['x_u'].to(device)
data_val['y_u'] = data_val['y_u'].to(device)
data_val['x_f'] = data_val['x_f'].to(device)
data_val['y_f'] = data_val['y_f'].to(device)

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
n_params_single = 1

# sampling

params_hmc = util.sample_model_bpinns(nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L, step_size=step_size, burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, n_params_single=n_params_single, pde=pde, pinns=pinns, epochs=epochs)

pred_list, log_prob_list = util.predict_model_bpinns(nets, params_hmc, data_val, model_loss=model_loss, tau_priors=tau_priors, tau_likes=tau_likes, n_params_single = n_params_single, pde = pde)

print('\nExpected validation log probability: {:.3f}'.format(torch.stack(log_prob_list).mean()))

print('\nThe exact values of single parameters: {}'.format(exact_single))
params_single = torch.stack(params_hmc)[:,:n_params_single].cpu().numpy()
print('The means of single parameters: {}'.format(np.exp(params_single).mean(0)))
print('The variances of single parameters: {}'.format(np.exp(params_single).std(0)))

pred_list_u = pred_list[0].cpu().numpy()
pred_list_f = pred_list[1].cpu().numpy()

# plot

extent = [lb,ub,lb,ub]
x_u = data['x_u'].cpu().numpy()
x_f = data['x_f'].cpu().numpy()

pred_mean_u = pred_list_u.mean(0).reshape(N_val,N_val)
plt.figure(figsize=(5,5))
plt.imshow(pred_mean_u, extent=extent)
plt.colorbar()
plt.plot(x_u[:,0],x_u[:,1],'ko', markerfacecolor='none')
plt.show()

pred_mean_f = pred_list_f.mean(0).reshape(N_val,N_val)
plt.figure(figsize=(5,5))
plt.imshow(pred_mean_f, extent=extent)
plt.colorbar()
plt.plot(x_f[:,0],x_f[:,1],'kx', markerfacecolor='none')
plt.show()
# %%
