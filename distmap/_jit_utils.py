"""Utility functions for TorchScript"""
# stdlib
import os
from typing import List
# dependencies
import torch
from torch import Tensor
# internals
from ._utils import torch_version


IS_JITSCRIPT_ACTIVATED = int(os.environ.get('PYTORCH_JIT', '1'))
IS_JITSCRIPT_DEPRECATED = torch_version('>=', (2, 10))


if IS_JITSCRIPT_DEPRECATED:
    def jitscript(func):
        return func
else:
    jitscript = torch.jit.script


@jitscript
def list_reverse_int(x: List[int]) -> List[int]:
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]


@jitscript
def movedim1(x, source: int, destination: int):
    dim = x.dim()
    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = [d for d in range(dim)]
    permutation = permutation[:source] + permutation[source+1:]
    permutation = permutation[:destination] + [source] + permutation[destination:]
    return x.permute(permutation)


@jitscript
def square(x):
    return x * x


if torch_version('>=', (1, 10)):
    # torch >= 1.10
    # -> use `indexing` keyword

    if IS_JITSCRIPT_DEPRECATED or not IS_JITSCRIPT_ACTIVATED:
        # JIT deactivated -> torch.meshgrid takes an unpacked list of tensors

        @jitscript
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(*tensors, indexing='ij'))

        @jitscript
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(*tensors, indexing='xy'))

    else:
        # JIT activated -> torch.meshgrid takes a packed list of tensors

        @jitscript
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors, indexing='ij'))

        @jitscript
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors, indexing='xy'))

else:
    # torch < 1.10
    # -> implement "xy" mode manually

    if IS_JITSCRIPT_DEPRECATED or not IS_JITSCRIPT_ACTIVATED:
        # JIT deactivated -> torch.meshgrid takes an unpacked list of tensors

        @jitscript
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors))

        @jitscript
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            grid = list(torch.meshgrid(*tensors))
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid

    else:
        # JIT activated -> torch.meshgrid takes a packed list of tensors

        @jitscript
        def meshgrid_list_ij(tensors: List[Tensor]) -> List[Tensor]:
            return list(torch.meshgrid(tensors))

        @jitscript
        def meshgrid_list_xy(tensors: List[Tensor]) -> List[Tensor]:
            grid = list(torch.meshgrid(tensors))
            if len(grid) > 1:
                grid[0] = grid[0].transpose(0, 1)
                grid[1] = grid[1].transpose(0, 1)
            return grid
