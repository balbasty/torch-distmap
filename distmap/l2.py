__all__ = ['euclidean_distance_transform', 'euclidean_signed_transform']
# dependencies
import torch
# internals
from . import _jitfields, backend
from ._utils import make_vector
from ._l1 import l1dt_1d_
from ._l2 import edt_1d


def euclidean_distance_transform(x, ndim=None, vx=1):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor. Zeros will stay zero, and the distance will
        be propagated into nonzero voxels.
    ndim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if backend.jitfields and _jitfields.available:
        return _jitfields.euclidean_distance_transform(x, ndim, vx)

    dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    ndim = ndim or x.dim()
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    x = l1dt_1d_(x, -ndim, vx[0])
    x *= x
    for d, w in zip(range(1, ndim), vx[1:]):
        x = edt_1d(x, d - ndim, w)
    if hasattr(x, 'sqrt_'):
        x.sqrt_()
    else:
        x = x ** 0.5
    return x


def euclidean_signed_transform(x, ndim=None, vx=1):
    """Compute the signed Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor.
        A negative distance will propagate into zero voxels and
        a positive distance will propagate into nonzero voxels.
    ndim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Signed distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if x.dtype is not torch.bool:
        x = x > 0
    d = euclidean_distance_transform(x, ndim, vx)
    d -= euclidean_distance_transform(~x, ndim, vx)
    return d
