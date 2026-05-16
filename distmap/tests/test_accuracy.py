import pytest
import torch
from ..l1 import l1_signed_transform
from ..l2 import euclidean_signed_transform
from .utils import make_ndsphere, make_cartesian_grid


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_accuracy_l1(ndim):
    x = make_ndsphere(ndim, 8)
    g = make_cartesian_grid(x.shape)
    y = l1_signed_transform(x)
    d = (g.reshape(-1, 1, ndim) - g.reshape(1, -1, ndim)).abs().sum(-1)
    dp = d[:, (~x).flatten()].min(1).values.reshape(x.shape).to(y)
    dm = d[:, x.flatten()].min(1).values.reshape(x.shape).to(y)
    d = torch.where(x, dp, -dm)
    assert torch.allclose(y, d)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_accuracy_l2(ndim):
    x = make_ndsphere(ndim, 8)
    g = make_cartesian_grid(x.shape)
    y = euclidean_signed_transform(x)
    d = ((g.reshape(-1, 1, ndim) - g.reshape(1, -1, ndim)) ** 2).sum(-1) ** 0.5
    dp = d[:, (~x).flatten()].min(1).values.reshape(x.shape).to(y)
    dm = d[:, x.flatten()].min(1).values.reshape(x.shape).to(y)
    d = torch.where(x, dp, -dm)
    assert torch.allclose(y, d)
