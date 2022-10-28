import torch
from .jit_utils import movedim1, square
Tensor = torch.Tensor


if hasattr(torch, 'true_divide'):
    _true_div = torch.true_divide
else:
    _true_div = torch.div


@torch.jit.script
def edt_1d_fillin(f, v, z, w2: float = 1.):
    # process along the first dimension
    #
    # f: input function
    # v: locations of parabolas in lower envelope
    # z: location of boundaries between parabolas

    k = f.new_zeros(f.shape[1:], dtype=torch.long)
    d = torch.empty_like(f)
    for q in range(len(f)):

        zk = z.gather(0, k[None] + 1)[0]
        mask = zk < q

        while mask.any():
            k = k.add_(mask)
            zk = z.gather(0, k[None] + 1)[0]
            mask = zk < q

        vk = v.gather(0, k[None])[0]
        fvk = f.gather(0, vk[None])[0]
        d[q] = w2 * square(q - vk) + fvk

    return d


@torch.jit.script
def edt_1d_intersection(f, v, z, k, q: int, w2: float = 1.):
    vk = v.gather(0, k[None])[0]
    fvk = f.gather(0, vk[None])[0]
    fq = f[q]
    a, b = w2 * (q - vk), q + vk
    s = _true_div((fq - fvk) + a * b, 2 * a)
    zk = z.gather(0, k[None])[0]
    mask = (k > 0) & (s <= zk)
    return s, mask


@torch.jit.script
def edt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 1 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """

    if f.shape[dim] == 1:
        return f

    w = w * w                                        # unit length (squared)
    f = movedim1(f, dim, 0)                          # input function
    k = f.new_zeros(f.shape[1:], dtype=torch.long)   # index of rightmost parabola in lower envelope
    v = f.new_zeros(f.shape, dtype=torch.long)       # locations of parabolas in lower envelope
    z = f.new_empty([len(f)+1] + list(f.shape[1:]))  # location of boundaries between parabolas

    # compute lower envelope
    z[0] = -float('inf')
    z[1] = float('inf')
    for q in range(1, len(f)):

        s, mask = edt_1d_intersection(f, v, z, k, q, w)
        while mask.any():
            k.add_(mask, alpha=-1)
            s, mask = edt_1d_intersection(f, v, z, k, q, w)

        s.masked_fill_(torch.isnan(s), -float('inf'))  # is this correct?

        k.add_(1)
        v.scatter_(0, k[None], q)
        z.scatter_(0, k[None], s[None])
        z.scatter_(0, k[None] + 1, float('inf'))

    # fill in values of distance transform
    d = edt_1d_fillin(f, v, z, w)
    d = movedim1(d, 0, dim)
    return d
