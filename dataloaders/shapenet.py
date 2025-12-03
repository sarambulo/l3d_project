from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch3d
import pytorch3d.datasets
from pathlib import Path

from pytorch3d.datasets import (
    R2N2,
    ShapeNetCore,
    collate_batched_meshes,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)

from pytorch3d.structures import Meshes, Volumes
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# add path for demo utils functions 
import sys
import os

class ShapeNetDataset(Dataset):
    def __init__(self, shapenet_dir: str = "./data/shapenet/", n_sample_points: int = 10000):
        self.shapenet_dir = shapenet_dir
        self.n_sample_points = n_sample_points

        self.shapenet_dataset = ShapeNetCore(self.shapenet_dir, version=2, load_textures=False)

    def __len__(self):
        return len(self.shapenet_dataset)

    def __getitem__(self, index):
        model = self.shapenet_dataset[index]
        verts = model['verts']
        faces = model['faces']

        mesh = Meshes(verts=[verts], faces=[faces])
        surface_points, normals = sample_points_from_meshes(mesh, num_samples=self.n_sample_points, return_normals=True) # (N, 3)
        surface_points = surface_points.squeeze(0)
        normals = normals.squeeze(0)
        min_vals = surface_points.min(dim=0)[0]  # (3,)
        max_vals = surface_points.max(dim=0)[0]  # (3,)
        
        center = center.mean(dim=0)
        surface_points = surface_points-center
        
        # scale = (max_vals - min_vals).max()
        norms = surface_points.norm(dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        surface_points = surface_points / norms

        points_normals = torch.cat([surface_points, normals], dim=-1)
        return points_normals, verts, faces # (N, 6), where first three are x, y and z and last three are normals along each axis

    def collate_fn(self, batch):
        points_list, verts_list, faces_list = zip(*batch)

        batch_points = torch.stack(points_list, dim=0)

        batch_vertices = pad_sequence(
            verts_list, 
            batch_first=True, 
            padding_value=0
        )

        batch_faces = pad_sequence(
            faces_list, 
            batch_first=True, 
            padding_value=-1
        )

        return batch_points, batch_vertices, batch_faces