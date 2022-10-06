from copy import deepcopy
import numpy as np
import torch

#SM implementation
def mutate_sm(model, states, mag=0.1, weight_clip=10.0, scaling_thresh=0.01, device='cpu'):
    """
    Mutate a model's parameters using SM-G-SUM [Lehman et al., 2018].

    Args:
        model: the model to mutate
        states: list of observations to calculate parameter sensitivities
        mag: magnitude of mutation
        weight_clip: maximum allowed change in weight
        scaling_thresh: minimum sensitivity

    Returns:
        mutated model (deepcopied; original model is unchanged) 
    """

    # evaluate model on states
    model = model.to(device)
    states = torch.tensor(np.array(states)).to(device)
    model_outputs = model(states)

    # calculate jacobian of derivatives of each output's sensitivity to each parameter
    num_classes = model_outputs.size()[1]
    num_params = count_parameters(model)
    jacobian = torch.zeros(num_classes, num_params)

    #do a backward pass for each output
    grad_output = torch.zeros(*model_outputs.size()).to(device)
    for i in range(num_classes):
        model.zero_grad()
        grad_output.zero_()
        grad_output[:, i] = 1.0

        model_outputs.backward(grad_output, retain_graph=True)
        jacobian[i] = extract_grad(model)

    # compute sensitivities
    scaling = torch.sqrt( (jacobian**2).sum(0) ).numpy()
    scaling[scaling == 0] = 1.0 # apply arbitrary scaling to parameters that don't affect output
    scaling[scaling < scaling_thresh] = scaling_thresh # apply minimum scaling
    
    # perturb the parameters
    params = extract_parameters(model)
    delta = torch.randn(*params.shape) * mag / scaling
    delta = torch.clip(delta, -weight_clip, weight_clip) # limit extreme weight changes for stability

    # inject perturbed parameters
    new_params = params + delta
    new_model = inject_parameters(model, new_params)
    return new_model

#function to return current pytorch gradient in same order as genome's flattened parameter vector
def extract_grad(model):
    return extract(model, lambda param: param.grad.data)

#function to grab current flattened neural network weights
def extract_parameters(model):
    return extract(model, lambda param: param.data)

def extract(model, extractor):
    tot_size = count_parameters(model)
    pvec = torch.zeros(tot_size)
    count = 0
    for param in model.parameters():
        sz = extractor(param).numel()
        pvec[count:count + sz] = param.data.flatten()
        count += sz 
    return pvec.clone()


#function to inject a flat vector of ANN parameters into the model's current neural network weights
def inject_parameters(model, pvec):
    model = deepcopy(model)
    count = 0

    for param in model.parameters():
        sz = torch.numel(param.data)
        param.data = pvec[count:count + sz].reshape(param.data.shape)
        count += sz

    return model

#count how many parameters are in the model
def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])
    