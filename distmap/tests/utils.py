import torch
from .._jit_utils import meshgrid_list_ij


def make_cartesian_grid(shape):
    return torch.stack(meshgrid_list_ij([torch.arange(s) for s in shape]), -1)


def make_ndsphere(ndim, shape=64):
    shape1 = shape
    shape = (shape,) * ndim
    x = make_cartesian_grid(shape)
    x = x.square().sum(-1).sqrt()
    x = (x < shape1/3)
    return x
