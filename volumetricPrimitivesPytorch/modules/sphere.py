import torch
import torch.nn as nn
from torch.autograd import Variable

class SphereSurface:
    def __init__(self, n_samples, normFactor="Surf"):
        self.n_samples = n_samples
        self.normFactor = normFactor

    def sample_points_sphere(self, shape):
        """
        shape: B x 1 x 3, representing sphere radius along each axis (or uniform radius)
        Returns: sampled_points B x n_samples x 3, imp_weights B x n_samples x 1
        """
        B = shape.size(0)
        # Sample points on unit sphere
        phi = torch.rand(B, self.n_samples, 1) * 2 * torch.pi
        costheta = torch.rand(B, self.n_samples, 1) * 2 - 1
        u = torch.rand(B, self.n_samples, 1)

        theta = torch.acos(costheta)
        r = u ** (1/3)  # uniform distribution in volume

        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)

        samples = torch.cat([x, y, z], dim=2)
        # scale by radius if you have non-uniform sphere axes
        samples = samples * shape

        # uniform importance weights
        imp_weights = torch.ones(B, self.n_samples, 1, device=samples.device)
        return samples, imp_weights