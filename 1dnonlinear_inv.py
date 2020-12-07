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
burn = 100
num_samples = 200
L = 100
layer_sizes = [1,16,16,1]
activation = torch.tanh
pde = True
pinns = False
epochs = 10000
tau_priors = 1/prior_std**2
tau_likes = 1/like_std**2

lb = -0.7
ub = 0.7
N_tr_u = 8
N_tr_f = 32
N_val = 100

# data

exact_single = np.array([0.7])

def u(x):
    return torch.sin(6*x)**3
def f(x):
    return 0.01 * (-108) * torch.sin(6*x)*(torch.sin(6*x)**2 - 2*torch.cos(6*x)**2) + exact_single[0] * torch.tanh(torch.sin(6*x)**3)

data = {}
data['x_u'] = torch.linspace(lb,ub,N_tr_u).view(-1,1)
data['y_u'] = u(data['x_u']) + torch.randn_like(data['x_u'])*like_std
data['x_f'] = torch.linspace(lb,ub,N_tr_f).view(-1,1)
data['y_f'] = f(data['x_f']) + torch.randn_like(data['x_f'])*like_std

data_val = {}
data_val['x_u'] = torch.linspace(lb,ub,N_val).view(-1,1)
data_val['y_u'] = u(data_val['x_u'])
data_val['x_f'] = torch.linspace(lb,ub,N_val).view(-1,1)
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

def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    x_u = data['x_u']
    y_u = data['y_u']
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data['x_f']
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u,x_f)[0]
    u_xx = gradients(u_x,x_f)[0]
    pred_f = 0.01*u_xx + torch.exp(params_single[0])*torch.tanh(u)
    y_f = data['y_f']
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u,pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output

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

x_val = data_val['x_u'].cpu().numpy()
u_val = data_val['y_u'].cpu().numpy()
f_val = data_val['y_f'].cpu().numpy()
x_u = data['x_u'].cpu().numpy()
y_u = data['y_u'].cpu().numpy()
x_f = data['x_f'].cpu().numpy()
y_f = data['y_f'].cpu().numpy()

plt.figure(figsize=(7,5))
plt.plot(x_val,u_val,'r-',label='Exact')
# plt.plot(x_val,pred_list_u.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val,pred_list_u.mean(0).squeeze().T, 'b-',alpha=0.9,label ='Mean')
plt.fill_between(x_val.reshape(-1), pred_list_u.mean(0).squeeze().T - 2*pred_list_u.std(0).squeeze().T, pred_list_u.mean(0).squeeze().T + 2*pred_list_u.std(0).squeeze().T, facecolor='b', alpha=0.2, label = '2 std')
plt.plot(x_u,y_u,'kx',markersize=5, label='Training data')
plt.xlim([lb,ub])
plt.legend(fontsize=10)
plt.show()

plt.figure(figsize=(7,5))
plt.plot(x_val,f_val,'r-',label='Exact')
# plt.plot(x_val,pred_list_f.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val,pred_list_f.mean(0).squeeze().T, 'b-',alpha=0.9,label ='Mean')
plt.fill_between(x_val.reshape(-1), pred_list_f.mean(0).squeeze().T - 2*pred_list_f.std(0).squeeze().T, pred_list_f.mean(0).squeeze().T + 2*pred_list_f.std(0).squeeze().T, facecolor='b', alpha=0.2, label = '2 std')
plt.plot(x_f,y_f,'kx',markersize=5, label='Training data')
plt.xlim([lb,ub])
plt.legend(fontsize=10)
plt.show()
# %%
