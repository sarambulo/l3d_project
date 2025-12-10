"""
Mesh to Dense Voxel Converter

Loads a mesh from an .obj file and converts it to a dense voxel grid
with the interior filled. Saves the result as a .npy file.

Usage:
    python mesh_to_voxel.py input.obj --resolution 128
    
Requirements:
    pip install trimesh numpy scipy
"""

import numpy as np
from tqdm import tqdm
import trimesh
import argparse
from pathlib import Path
from pytorch3d.io import load_obj
import torch
import matplotlib.pyplot as plt

def mesh_to_inside_points(obj_path, resolution=128):
    """
    Convert a mesh to a set of inside points.
    
    Args:
        mesh_path: Path to .obj file
        resolution: Voxel grid resolution (grid will be resolution^3)
        
    Returns:
        points: (N, 3) numpy array
    """
    verts, faces_info, _ = load_obj(obj_path, load_textures=False)
    verts = verts.to(torch.float)
    faces = faces_info.verts_idx.to(torch.int)

    mesh = trimesh.Trimesh(verts, faces)
    
    # Ensure mesh has consistent winding and is watertight
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight. Results may be unexpected.")
        # Attempt to fix
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
    
    # Get mesh bounds
    bounds = mesh.bounds  # (2, 3) array: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    mesh_min = bounds[0]
    mesh_max = bounds[1]
    mesh_size = mesh_max - mesh_min
            
    # Create a 3D grid of points
    x = np.linspace(mesh_min[0], mesh_max[0], resolution)
    y = np.linspace(mesh_min[1], mesh_max[1], resolution)
    z = np.linspace(mesh_min[2], mesh_max[2], resolution)
    
    # Create meshgrid
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
    
    # Use ray casting to determine if points are inside
    inside_all = []
    N = len(points)
    BATCH_SIZE = 200
    for i in tqdm(np.arange(0, N, BATCH_SIZE), desc="Ray casting", leave=False):
        inside = mesh.contains(points[i:i+BATCH_SIZE])
        inside_all.append(inside)
    inside_array = np.concat(inside_all)
    
    # Count filled voxels
    filled_count = inside_array.astype(int).sum()
    total_count = len(points)
    fill_percentage =filled_count / total_count
    
    print(f"Interior points: {filled_count:,} / {total_count:,} ({fill_percentage:.2%})")

    # Return dense point cloud
    points = points[inside_array]
    
    return points


def main():
    parser = argparse.ArgumentParser(
        description='Convert meshes to interior points'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing .obj files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for .npy files'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=32,
        help='Grid resolution'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' not found")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .obj files
    obj_files = sorted(input_path.glob("*.obj"))
    if not obj_files:
        print(f"No .obj files found in '{input_path}'")
        return
    
    print(f"Found {len(obj_files)} .obj files in {input_path}")
    
    # Process each mesh
    for obj_file in tqdm(obj_files, desc="Converting meshes"):
        # Convert mesh to inside points
        points = mesh_to_inside_points(
            str(obj_file),
            resolution=args.resolution,
        )
        
        # Save as .npy with same filename
        output_npy = output_path / obj_file.stem
        output_npy = output_npy.with_suffix('.npy')
        np.save(output_npy, points)
        print(f"Saved {output_npy}")

if __name__ == "__main__":
    main()