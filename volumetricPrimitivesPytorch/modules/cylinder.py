import torch
import torch.nn as nn
from torch.autograd import Variable

class CylinderSurface:
    def __init__(self, n_samples, normFactor="Surf"):
        self.n_samples = n_samples
        self.normFactor = normFactor

    def sample_points_cylinder(self, shape):
        """
        shape: B x 1 x 3, representing (radius, radius, height)
        Returns: sampled_points B x n_samples x 3, imp_weights B x n_samples x 1
        """
        B = shape.size(0)
        radius = shape[:, :, 0:1]
        height = shape[:, :, 2:3]

        theta = torch.rand(B, self.n_samples, 1) * 2 * torch.pi
        z = torch.rand(B, self.n_samples, 1) * height - height/2
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        samples = torch.cat([x, y, z], dim=2)

        imp_weights = torch.ones(B, self.n_samples, 1, device=samples.device)
        return samples, imp_weights
