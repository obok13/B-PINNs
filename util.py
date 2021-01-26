import torch
import torch.nn as nn
import hamiltorch
import numpy as np
import time

def build_lists(models, n_params_single=None, tau_priors=None, tau_likes=0.1, pde = False):

    if n_params_single is not None:
        n_params = [n_params_single]
    else:
        n_params = []

    if isinstance(tau_priors,list) or tau_priors is None:
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
    if pde and build_tau_likes:
        tau_likes.append(tau_likes_elt)

    n_params = list(np.cumsum(n_params))

    return params_shape_list, params_flattened_list, n_params, tau_priors, tau_likes


def define_model_log_prob_bpinns(models, model_loss, data, tau_priors=None, tau_likes=None, predict=False, prior_scale = 1.0, n_params_single = None, pde = False):

    """This function defines the `log_prob_func` for torch nn.Modules. This will then be passed into the hamiltorch sampler. This is an important
    function for any work with Bayesian neural networks.

    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
    data : dictionary
        Training input output data of each model.
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    predict : bool
        Flag to set equal to `True` when used as part of `hamiltorch.predict_model`, otherwise set to False. This controls the number of objects to return.
    prior_scale : float
        Most relevant for splitting (otherwise leave as 1.0). The prior is divided by this value.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether it is pde or not.

    Returns
    -------
    function
        Returns a `log_prob_func`, which takes a 1-D torch.tensor of a length equal to the parameter dimension and returns a single value.

    """

    _, params_flattened_list, n_params, tau_priors, tau_likes = build_lists(models, n_params_single, tau_priors, tau_likes, pde)

    fmodel = []
    for model in models:
        fmodel.append(hamiltorch.util.make_functional(model))

    if tau_priors is not None:
        dist_list = []
        for tau in tau_priors:
            dist_list.append(torch.distributions.Normal(0, tau**-0.5))

    def log_prob_func(params):

        params_unflattened = []
        if n_params_single is not None:
            params_single = params[:n_params[0]]
            for i in range(len(models)):
                params_unflattened.append(hamiltorch.util.unflatten(models[i], params[n_params[i]:n_params[i+1]]))
        else:
            params_single = None
            for i in range(len(models)):
                if i == 0:
                    params_unflattened.append(hamiltorch.util.unflatten(models[i], params[:n_params[i]]))
                else:
                    params_unflattened.append(hamiltorch.util.unflatten(models[i] ,params[n_params[i-1]:n_params[i]]))

        l_prior = torch.zeros_like( params[0], requires_grad=True)
        if tau_priors is not None:
            i_prev = 0
            for index, dist in zip(params_flattened_list, dist_list):
                w = params[i_prev:index+i_prev]
                l_prior = dist.log_prob(w).sum() + l_prior
                i_prev += index

        def gradients(outputs, inputs):
            return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

        ll, output = model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single)

        if predict:
            return (ll + l_prior/prior_scale), output
        else:
            return (ll + l_prior/prior_scale)

    return log_prob_func


def sample_model_bpinns(models, data, model_loss, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, inv_mass=None, jitter=None, normalizing_const=1., softabs_const=None, explicit_binding_const=100, fixed_point_threshold=1e-5, fixed_point_max_iterations=1000, jitter_max_tries=10, sampler=hamiltorch.Sampler.HMC, integrator=hamiltorch.Integrator.IMPLICIT, metric=hamiltorch.Metric.HESSIAN, debug=False, tau_priors=None, tau_likes=None, store_on_GPU = True, desired_accept_rate=0.8, device = 'cpu', n_params_single = None, pde = False, pinns = False, epochs = 10000, params_init_val = None):

    """Sample weights from a NN model to perform inference. This function builds a `log_prob_func` from the torch.nn.Module and passes it to `hamiltorch.sample`.

    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    data : dictionary
        Training input output data of each model.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
    num_samples : int
        Sets the number of samples corresponding to the number of momentum resampling steps/the number of trajectories to sample.
    num_steps_per_sample : int
        The number of steps to take per trajector (often referred to as L).
    step_size : float
        Size of each step to take when doing the numerical integration.
    burn : int
        Number of samples to burn before collecting samples. Set to -1 for no burning of samples. This must be less than `num_samples` as `num_samples` subsumes `burn`.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.
    jitter : float
        Jitter is often added to the diagonal to the metric tensor to ensure it can be inverted. `jitter` is a float corresponding to scale of random draws from a uniform distribution.
    normalizing_const : float
        This constant is currently set to 1.0 and might be removed in future versions as it plays no immediate role.
    softabs_const : float
        Controls the "filtering" strength of the negative eigenvalues. Large values -> absolute value. See Betancourt 2013.
    explicit_binding_const : float
        Only relevant to Explicit RMHMC. Corresponds to the binding term in Cobb et al. 2019.
    fixed_point_threshold : float
        Only relevant for Implicit RMHMC. Sets the convergence threshold for 'breaking out' of the while loop for the generalised leapfrog.
    fixed_point_max_iterations : int
        Only relevant for Implicit RMHMC. Limits the number of fixed point iterations in the generalised leapforg.
    jitter_max_tries : float
        Only relevant for RMHMC. Number of attempts at resampling the jitter for the Fisher Information before raising a LogProbError.
    sampler : Sampler
        Sets the type of sampler that is being used for HMC: Choice {Sampler.HMC, Sampler.RMHMC, Sampler.HMC_NUTS}.
    integrator : Integrator
        Sets the type of integrator to be used for the leapfrog: Choice {Integrator.EXPLICIT, Integrator.IMPLICIT, Integrator.SPLITTING, Integrator.SPLITTING_RAND, Integrator.SPLITTING_KMID}.
    metric : Metric
        Determines the metric to be used for RMHMC. E.g. default is the Hessian hamiltorch.Metric.HESSIAN.
    debug : {0, 1, 2}
        Debug mode can take 3 options. Setting debug = 0 (default) allows the sampler to run as normal. Setting debug = 1 prints both the old and new Hamiltonians per iteration, and also prints the convergence values when using the generalised leapfrog (IMPLICIT RMHMC). Setting debug = 2, ensures an additional float is returned corresponding to the acceptance rate or the adapted step size (depending if NUTS is used.)
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    store_on_GPU : bool
        Option that determines whether to keep samples in GPU memory. It runs fast when set to TRUE but may run out of memory unless set to FALSE.
    desired_accept_rate : float
        Only relevant for NUTS. Sets the ideal acceptance rate that the NUTS will converge to.
    device : name of device, or {'cpu', 'cuda'}
        The device to run on.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether it is pde or not.
    pinns : bool
        If this is true then `sample_model_bpinns` finds the MAP of the posterior instead of samples, which is the result of PINNs.
    epochs : int
        Determines the number of epochs when pinns is true

    Returns
    -------
    param_samples : list of torch.tensor(s)
        A list of parameter samples. The full trajectory will be returned such that selecting the proposed params requires indexing [1::L] to remove params_innit and select the end of the trajectories.
    step_size : float, optional
        Only returned when debug = 2 and using NUTS. This is the final adapted step size.
    acc_rate : float, optional
        Only returned when debug = 2 and not using NUTS. This is the acceptance rate.

    """

    if n_params_single is not None:
        params_init = torch.zeros([n_params_single]).to(device)
    else:
        params_init = torch.Tensor([]).to(device)

    for model in models:
        params_init_net = hamiltorch.util.flatten(model).to(device).clone()
        params_init = torch.cat((params_init,params_init_net))

    # params_init = torch.randn_like(params_init)
    if params_init_val is not None:
        params_init = params_init_val

    log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, n_params_single = n_params_single, pde = pde)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    if pinns:
        params = params_init.clone().detach().requires_grad_()
        optimizer = torch.optim.Adam([params], lr=step_size)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = - log_prob_func(params)
            loss.backward()
            optimizer.step()

            if epoch%100==0:
                print('Epoch: %d, loss: %.6f, time: %.2f' % (epoch, loss.detach().item(), time.time() - start_time))

        if not store_on_GPU:
            ret_params = [params.clone().detach().cpu()]
        else:
            ret_params = [params.clone()]

        return list(map(lambda t: t.detach(), ret_params))

    else:
        return hamiltorch.sample(log_prob_func, params_init, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, jitter=jitter, inv_mass=inv_mass, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, jitter_max_tries=jitter_max_tries, sampler=sampler, integrator=integrator, metric=metric, debug=debug, desired_accept_rate=desired_accept_rate, store_on_GPU = store_on_GPU)

def predict_model_bpinns(models, samples, data, model_loss, tau_priors=None, tau_likes=None, n_params_single = None, pde = False):

    """Function used to make predictions given model samples. Note that either a data loader can be passed in, or two tensors (x,y) but make sure
    not to pass in both.

    Parameters
    ----------
    models : list of torch.nn.Module(s)
        This is the list of torch neural network models, which will be used when performing inference.
    samples : list of torch.tensor(s)
        A list, where each element is a torch.tensor of shape (D,), where D is the number of parameters of the model. The length of the list is given by the number of samples, S.
    data : dictionary
        Training input output data of each model.
    model_loss : function
        This determines the likelihood to be used for the model. You can customize this function in main code.
    tau_priors: float or list of float(s)
        Determines the stds of gaussian priors for parameters. If this is None then the priors become uniform distribution. If this is float then it becomes std of priors for all parameters. If this is a list then each element of the list becomes std of priors for [1st single parameter, 2nd single parameter,..., weights of 1st hidden layer, bias of 1st hidden layer, weights of 2nd hidden layer, bias of 2nd hidden layer,...]
    tau_likes: float or list of float(s)
        Data are assumed to be collected with gaussian noise and tau_likes determines the std of noise. If this is float then it becomes std of noise for all data. If this is a list then each element of the list becomes std of noise for each element of the list of models.
    n_params_single : int
        The number of single parameters that have to be inferred.
    pde : bool
        Determines whether it is pde or not.

    Returns
    -------
    predictions : torch.tensor
        Output of the model of shape (S,N,O), where S is the number of samples, N is the number of data points, and O is the output shape of the model.
    pred_log_prob_list : list
        List of log probability values for each sample. The length of the list is S.

    """

    if pde:

        log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True, n_params_single = n_params_single, pde = pde)

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

        for i in range(len(pred_list)):
            pred_list[i] = torch.stack(pred_list[i])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_list, pred_log_prob_list

    else:
        with torch.no_grad():

            log_prob_func = define_model_log_prob_bpinns(models, model_loss, data, tau_priors, tau_likes, predict=True, n_params_single = n_params_single, pde = pde)

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

            for i in range(len(pred_list)):
                pred_list[i] = torch.stack(pred_list[i])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return pred_list, pred_log_prob_list

