try:
    import jitfields.distance
    _jitfields = jitfields
    available = True
except (ImportError, ModuleNotFoundError):
    _jitfields = None
    available = False


def euclidean_distance_transform(x, ndim=None, vx=1):
    return _jitfields.distance.euclidean_distance_transform(x, ndim, vx)


def l1_distance_transform(x, ndim=None, vx=1):
    return _jitfields.distance.l1_distance_transform(x, ndim, vx)
