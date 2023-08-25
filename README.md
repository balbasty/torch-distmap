# torch-distmap

Euclidean distance transform in PyTorch.

This is an implementation of the algorithm from the paper
    
> [**"Distance Transforms of Sampled Functions"**](https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf) <br />
> Pedro F. Felzenszwalb & Daniel P. Huttenlocher <br />
> _Theory of Computing_ (2012)

Although it is in PyTorch, our implementation performs loops across 
voxels and hence quite slow. Moreover, it takes masks as an input 
and therefore does not allow backpropagation.

## Installation

### Dependency

- `torch >= 1.3`

### Conda

```shell
conda install torch-distmap -c balbasty -c pytorch
```

### Pip

```shell
pip install torch-distmap
```

## API

```python
euclidean_distance_transform(x, ndim=None, vx=1)
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
"""
```

```python
euclidean_signed_transform(x, ndim=None, vx=1)
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
"""
```

```python
l1_distance_transform(x, ndim=None, vx=1)
"""Compute the L1 distance transform of a binary image

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
"""
```

```python
l1_signed_transform(x, ndim=None, vx=1)
"""Compute the signed L1 distance transform of a binary image

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
"""
```

## Related packages

- [edt](https://github.com/seung-lab/euclidean-distance-transform-3d) : 
  a very fast CPU implementation of the same algorithm, written in C.


- [scipy.ndimage.distance_transform_edt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html) :
reference implementation, written in C, based on the paper
> **"A linear time algorithm for computing exact euclidean distance 
> transforms of binary images in arbitrary dimensions"** <br />
> C. R. Maurer,  Jr., R. Qi, V. Raghavan <br />
> IEEE Trans. PAMI 25, 265-270, (2003) <br />
