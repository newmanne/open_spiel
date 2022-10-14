from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.autograd.gradcheck import zero_gradients
from torch.nn import Parameter

#SM implementation
def mutate_sm(model, states, mag=0.1, weight_clip=10.0, scaling_thresh=0.01):
    params = extract_parameters(model)

    #experience in this domain = the classification *input* patterns  
    experience_states = Variable(torch.from_numpy(states), requires_grad=False)

    #old_policy in this domain = the outputs this model generated before perturbation
    old_policy = model(experience_states)

    num_classes = old_policy.size()[1]

    #initial perturbation
    delta = np.random.randn(*params.shape).astype(np.float32) * mag

    tot_size = model.count_parameters()

    #we want to calculate a jacobian of derivatives of each output's sensitivity to each parameter
    jacobian = torch.zeros(num_classes, tot_size)
    grad_output = torch.zeros(*old_policy.size())

    #do a backward pass for each output
    for i in range(num_classes):
        model.zero_grad()
        grad_output.zero_()
        grad_output[:, i] = 1.0

        old_policy.backward(grad_output, retain_graph=True)
        jacobian[i] = torch.from_numpy(model.extract_grad())

    #summed gradients sensitivity
    scaling = torch.sqrt( (jacobian**2).sum(0) )

    scaling = scaling.numpy()
    
    scaling[scaling == 0] = 1.0
    scaling[scaling < scaling_thresh] = scaling_thresh
    
    delta /= scaling

    #limit extreme weight changes for stability
    final_delta = np.clip(delta, -weight_clip, weight_clip)  #as 1.0

    #generate new parameter vector
    new_params = params + final_delta

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
    pvec = np.zeros(tot_size, np.float32)
    count = 0
    for param in model.parameters():
        sz = extractor(param).numel()
        pvec[count:count + sz] = param.data.numpy().flatten()
        count += sz 
    return pvec.copy()


#function to inject a flat vector of ANN parameters into the model's current neural network weights
def inject_parameters(model, pvec):
    model = deepcopy(model)
    count = 0

    for param in model.parameters():
        sz = torch.numel(param.data)
        raw = pvec[count:count + sz]
        reshaped = raw.reshape(param.data.shape)
        param.data = torch.from_numpy(reshaped)
        count += sz

    return model

#count how many parameters are in the model
def count_parameters(model):
    return sum([p.numel() for p in model.parameters()])
    