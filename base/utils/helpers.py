import numpy as np
import random
import torch

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'softmax':
        return torch.nn.Softmax(dim=1)
    elif activation == 'log_softmax':
        return torch.nn.LogSoftmax(dim=1)
    else:
        raise ValueError('Activation function not supported')