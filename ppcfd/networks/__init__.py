import operator
from functools import reduce



try:
    import open3d
    from .GNOFNOGNO import GNOFNOGNO
    from .GNOFNOGNO import GNOFNOGNOAhmed
except ImportError:
    print("open3d are not imported")
import sys

from .KAN import KAN
from .MLP import MLP
from .LRQSolver import LRQSolver
from .Transolver import Transolver
from .RegPointNet import RegPointNet
from .GeomDeepONet import GeomDeepONet
from .Transformer import Point_Transformer
from .MLP import MLP_with_boundary_condition
from .ConvUNet2 import UNet3DWithSamplePoints
from .Transolver_0513_0 import Transolver_0513_0

__all__ = [
    "Transolver",
    "GNOFNOGNO",
    "GNOFNOGNOAhmed",
    "KAN",
    "UNet3DWithSamplePoints",
    "MLP_with_boundary_condition",
    "Transolver_0513_0",
    "RegPointNet",
    "MLP",
    "Point_Transformer",
    "GeomDeepONet",
    "LRQSolver",
]


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    return c
