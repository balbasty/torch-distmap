try:
    import jitfields
    available = True
except (ImportError, ModuleNotFoundError):
    jitfields = None
    available = False
from .utils import make_list
import torch


def euclidean_distance_transform(x, ndim=None, vx=1):
    return jitfields.euclidean_distance_transform(x, ndim, vx)


def l1_distance_transform(x, ndim=None, vx=1):
    return jitfields.l1_distance_transform(x, ndim, vx)
