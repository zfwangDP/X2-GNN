import torch

def _standardize(kernel):
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel

def he_orthogonal_init(tensor):
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor

def Glorot_Ortho_(tensor, scale = 2.0):
    tensor = torch.nn.init.orthogonal_(tensor)
    with torch.no_grad():
        assert len(tensor.shape) == 2
        tensor.data *= torch.sqrt(scale/((tensor.size()[0]+tensor.size()[1])*tensor.var()))
    
    return tensor