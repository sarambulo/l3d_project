import torch
from torch.autograd import Variable
import pytorch3d.ops
from pytorch3d.structures import Volumes

from modules.cuboid import CuboidSurface
from modules.transformer import rigidPointsTransform
from torch import nn
from pytorch3d.loss import chamfer_distance