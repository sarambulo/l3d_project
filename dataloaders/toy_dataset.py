from json import load
from typing import Literal
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch3d
from pytorch3d.io import load_obj
from pathlib import Path
from pytorch3d.renderer import (
    TexturesVertex,
)

from pytorch3d.structures import Meshes, Volumes
from pytorch3d.ops import sample_points_from_meshes
from torch.nn.utils.rnn import pad_sequence

from utils import interior_points

class ToyDataset(Dataset):
    def __init__(self, data_dir: str = "./data/toy_dataset/", n_sample_points: int = 10000):
        self.data_dir = data_dir
        self.n_sample_points = n_sample_points

        # Collect all .obj files
        data_path = Path(self.data_dir)
        if data_path.exists():
            self.obj_files = sorted(data_path.glob("*.obj"))
            self.interior_points_files = sorted(data_path.glob("*.npy"))
        else:
            raise ValueError()

        # Load all objects
        self.objects = [
            self._load_mesh(str(file))
            for file in self.obj_files
        ]

        # Load all interior points
        self.interior_points = [
            torch.tensor(np.load(file), dtype=torch.float)
            for file in self.interior_points_files
        ]

    def _load_mesh(self, filename: str):
        verts, faces_info, aux = load_obj(filename, load_textures=False)
        verts = verts.to(torch.float)
        faces = faces_info.verts_idx.to(torch.int)

        center = verts.mean(dim=0, keepdim=True) # (N, 3) -> (N, 1)
        scale = verts.norm(dim=-1).max() # (N, 3) -> (N,) -> ()
        verts = (verts-center)/scale

        mesh = Meshes(verts=[verts], faces=[faces])
        points, normals = sample_points_from_meshes(
            mesh, num_samples=self.n_sample_points, return_normals=True
        ) # (N, 3)
        points = points.squeeze(0) # (N, 3)
        normals = normals.squeeze(0) # (N, 3)
       

        points_with_normals = torch.cat([points, normals], dim=-1) # (N, 6)

        return points_with_normals, verts, faces

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, index):
        points_with_normals, verts, faces = self.objects[index]
        interior_points = self.interior_points[index]
        
        return points_with_normals, verts, faces, interior_points # (N, 6), where first three are x, y and z and last three are normals along each axis

    def collate_fn(self, batch):
        points_list, verts_list, faces_list, interior_points_list = zip(*batch)

        batch_points = torch.stack(points_list, dim=0) # (B, N, 6)

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

        batch_interior_points = list(interior_points_list) # [(N, 3)]

        return batch_points, batch_vertices, batch_faces, batch_interior_points