import torch
import torch.nn as nn
import hamiltorch
import numpy as np

def build_lists(models, n_params_single=None, tau_priors=1., tau_likes=0.1, pde = False):

    if n_params_single is not None:
        n_params = [n_params_single]
    else:
        n_params = []

    if isinstance(tau_priors,list):
        build_tau_priors = False
    else:
        build_tau_priors = True
        tau_priors_elt = tau_priors
        tau_priors = []

    if isinstance(tau_likes,list):
        build_tau_likes = False
    else:
        build_tau_likes = True
        tau_likes_elt = tau_likes
        tau_likes = []

    params_shape_list = []
    params_flattened_list = []

    if build_tau_priors and n_params_single is not None:
        for _ in range(n_params_single):
            params_flattened_list.append(1)
            params_shape_list.append(1)
            tau_priors.append(tau_priors_elt)

    for model in models:
        n_params.append(hamiltorch.util.flatten(model).shape[0])
        if build_tau_likes:
            tau_likes.append(tau_likes_elt)
        for weights in model.parameters():
            params_shape_list.append(weights.shape)
            params_flattened_list.append(weights.nelement())
            if build_tau_priors:
                tau_priors.append(tau_priors_elt)

    # if we deal with pde then we also have data of residual
    if pde:
        tau_likes.append(tau_likes_elt)

    n_params = list(np.cumsum(n_params))

    return params_shape_list, params_flattened_list, n_params, tau_priors, tau_likes


def define_model_log_prob_bpinns(models, model_loss, data, tau_priors=None, tau_likes=None, predict=False, prior_scale = 1.0, device = 'cpu', n_params_single = None, pde = False):

    _, params_flattened_list, n_params, tau_priors, tau_likes = build_lists(models, n_params_single, tau_priors, tau_likes, pde)

    if len(tau_likes) == 1:
        tau_likes = tau_likes[0] # If the length is 1, then release the variable from list.

    if len(models) == 1:
        fmodel = hamiltorch.util.make_functional(models[0])
    else:
        fmodel = []
        for model in models:
            fmodel.append(hamiltorch.util.make_functional(model))

    dist_list = []
    for tau in tau_priors:
        dist_list.append(torch.distributions.Normal(0, tau**-0.5))

    def log_prob_func(params):

        if len(models) == 1:
            if n_params_single is not None:
                params_single = params[:n_params[0]]
                params_unflattened = hamiltorch.util.unflatten(models[0], params[n_params[0]:])
            else:
                params_unflattened = hamiltorch.util.unflatten(models[0], params)
        else:
            params_unflattened = []
            if n_params_single is not None:
                params_single = params[:n_params[0]]
                for i in range(len(models)):
                    params_unflattened.append(hamiltorch.util.unflatten(models[0], params[n_params[i]:n_params[i+1]]))
            else:
                for i in range(len(models)):
                    if i == 0:
                        params_unflattened.append(hamiltorch.util.unflatten(models[0], params[:n_params[i]]))
                    else:
                        params_unflattened.append(hamiltorch.util.unflatten(models[0],params[n_params[i-1]:n_params[i]]))

        i_prev = 0
        l_prior = torch.zeros_like( params[0], requires_grad=True) # Set l2_reg to be on the same device as params
        for index, dist in zip(params_flattened_list, dist_list):
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        # Sample prior if no data
        if data is None:
            return l_prior/prior_scale

        def gradients(outputs, inputs):
            return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

        if model_loss is 'prior':
            x = data['x'].to(device)
            pred = fmodel(x, params=params_unflattened)
            ll = 0
            output = [pred]

            if torch.cuda.is_available():
                del x
                torch.cuda.empty_cache()
 
        elif model_loss is '1dregression':
            x = data['x'].to(device)
            y = data['y'].to(device)
            pred = fmodel(x, params=params_unflattened)
            ll = - 0.5 * tau_likes * ((pred - y) ** 2).sum(0)
            output = [pred]

            if torch.cuda.is_available():
                del x, y
                torch.cuda.empty_cache()
 
        elif model_loss is '1dpoisson':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            u_x = gradients(u,x_f)[0]
            u_xx = gradients(u_x,x_f)[0]
            pred_f = 0.01*u_xx
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '1dporous':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            u_x = gradients(u,x_f)[0]
            u_xx = gradients(u_x,x_f)[0]
            pred_f = - 1e-3/0.4*u_xx + u
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '1dnonlinear':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            u_x = gradients(u,x_f)[0]
            u_xx = gradients(u_x,x_f)[0]
            pred_f = 0.01*u_xx + 0.7*torch.tanh(u)
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '1dnonlinear_inv':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            u_x = gradients(u,x_f)[0]
            u_xx = gradients(u_x,x_f)[0]
            pred_f = 0.01*u_xx + torch.exp(params_single[0])*torch.tanh(u)
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_xx, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '2dnonlinear':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            Du = gradients(u, x_f)[0]
            u_x, u_y = Du[:,0:1], Du[:,1:2]
            u_xx = gradients(u_x, x_f)[0][:,0:1]
            u_yy = gradients(u_y, x_f)[0][:,1:2]
            pred_f = 0.01*(u_xx+u_yy) + u*(u**2-1)
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_y, u_xx, u_yy, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '2dnonlinear_inv':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel(x_u, params=params_unflattened)
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel(x_f, params=params_unflattened)
            Du = gradients(u, x_f)[0]
            u_x, u_y = Du[:,0:1], Du[:,1:2]
            u_xx = gradients(u_x, x_f)[0][:,0:1]
            u_yy = gradients(u_y, x_f)[0][:,1:2]
            pred_f = 0.01*(u_xx+u_yy) + torch.exp(params_single[0])*u**2
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_y, u_xx, u_yy, pred_u, pred_f
                torch.cuda.empty_cache()
 
        elif model_loss is '1dinferk':
            x_u = data['x_u'].to(device)
            y_u = data['y_u'].to(device)
            pred_u = fmodel[0](x_u, params=params_unflattened[0])
            ll = - 0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
            x_k = data['x_k'].to(device)
            y_k = data['y_k'].to(device)
            pred_k = fmodel[1](x_k, params=params_unflattened[1])
            ll = ll - 0.5 * tau_likes[1] * ((pred_k - y_k) ** 2).sum(0)
            x_f = data['x_f'].to(device)
            x_f = x_f.detach().requires_grad_()
            u = fmodel[0](x_f, params=params_unflattened[0])
            u_x = gradients(u,x_f)[0]
            u_xx = gradients(u_x,x_f)[0]
            k = fmodel[1](x_f, params=params_unflattened[1])
            pred_f = 0.01*u_xx + k*u + torch.exp(params_single[0])*u**2 + torch.exp(params_single[1])*torch.tanh(u)
            y_f = data['y_f'].to(device)
            ll = ll - 0.5 * tau_likes[2] * ((pred_f - y_f) ** 2).sum(0)
            output = [pred_u,pred_k,pred_f]

            if torch.cuda.is_available():
                del x_u, y_u, x_f, y_f, u, u_x, u_xx, k, pred_u, pred_k, pred_f
                torch.cuda.empty_cache()
 
        else:
            raise NotImplementedError()

        if predict:
            return (ll + l_prior/prior_scale), output
        else:
            return (ll + l_prior/prior_scale)

    return log_prob_func


def sample_model_bpinns(models, data, model_loss, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, inv_mass=None, jitter=None, normalizing_const=1., softabs_const=None, explicit_binding_const=100, fixed_point_threshold=1e-5, fixed_point_max_iterations=1000, jitter_max_tries=10, sampler=hamiltorch.Sampler.HMC, integrator=hamiltorch.Integrator.IMPLICIT, metric=hamiltorch.Metric.HESSIAN, debug=False, tau_priors=None, tau_likes=None, store_on_GPU = True, desired_accept_rate=0.8, device = 'cpu', n_params_single = None, pde = False, pinns = False, epochs = 10000):

    if n_params_single is not None:
        params_init = torch.zeros([n_params_single]).to(device)
    else:
        params_init = torch.Tensor([]).to(device)

    for model in models:
        params_init_net = hamiltorch.util.flatten(model).to(device).clone()
        params_init = torch.cat((params_init,params_init_net))

    # params_init = torch.randn_like(params_init)
    print('Parameter size: ', params_init.shape[0])

    log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, device = device, n_params_single = n_params_single, pde = pde)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if pinns:
        params = params_init.clone().detach().requires_grad_()
        optimizer = torch.optim.Adam([params], lr=step_size)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = - log_prob_func(params)
            loss.backward()
            optimizer.step()

            if epoch%100==0:
                print('[Epoch]'+str(epoch)+', loss:'+str(loss.detach()))

        if not store_on_GPU:
            ret_params = [params.clone().detach().cpu()]
        else:
            ret_params = [params.clone()]

        return list(map(lambda t: t.detach(), ret_params))

    else:
        return hamiltorch.sample(log_prob_func, params_init, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, jitter=jitter, inv_mass=inv_mass, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, jitter_max_tries=jitter_max_tries, sampler=sampler, integrator=integrator, metric=metric, debug=debug, desired_accept_rate=desired_accept_rate, store_on_GPU = store_on_GPU)

def predict_model_bpinns(models, samples, data, model_loss, tau_priors=None, tau_likes=None, n_params_single = None, pde = False):

    if pde:

        log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True, device = samples[0].device, n_params_single = n_params_single, pde = pde)

        pred_log_prob_list = []
        pred_list = []
        _, pred = log_prob_func(samples[0])
        for i in range(len(pred)):
            pred_list.append([])

        for s in samples:
            lp, pred = log_prob_func(s)
            pred_log_prob_list.append(lp.detach()) # Side effect is to update weights to be s
            for i in range(len(pred_list)):
                pred_list[i].append(pred[i].detach())
        
        _, pred = log_prob_func(samples[0])
        for i in range(len(pred_list)):
            pred_list[i] = torch.stack(pred_list[i])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_list, pred_log_prob_list

    else:
        with torch.no_grad():

            log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True, device = samples[0].device, n_params_single = n_params_single, pde = pde)

            pred_log_prob_list = []
            pred_list = []
            _, pred = log_prob_func(samples[0])
            for i in range(len(pred)):
                pred_list.append([])

            for s in samples:
                lp, pred = log_prob_func(s)
                pred_log_prob_list.append(lp.detach()) # Side effect is to update weights to be s
                for i in range(len(pred_list)):
                    pred_list[i].append(pred[i].detach())
            
            _, pred = log_prob_func(samples[0])
            for i in range(len(pred_list)):
                pred_list[i] = torch.stack(pred_list[i])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return pred_list, pred_log_prob_list

