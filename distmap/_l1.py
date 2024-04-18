import torch
from .jit_utils import movedim1, list_reverse_int
from typing import List
Tensor = torch.Tensor


@torch.jit.script
def l1dt_1d_(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if f.shape[dim] == 1:
        return f

    f = movedim1(f, dim, 0)

    for q in range(1, len(f)):
        f[q] = torch.min(f[q], f[q-1] + w)
    rng: List[int] = [e for e in range(len(f)-1)]
    for q in list_reverse_int(rng):
        f[q] = torch.min(f[q], f[q+1] + w)

    f = movedim1(f, 0, dim)
    return f


@torch.jit.script
def l1dt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    return l1dt_1d_(f.clone(), dim, w)
