"""
Visualize voxelized mesh as rotating 3D GIF.

Usage:
    python visualize_voxels.py --mesh_path toy_dataset/sphere_with_hole.obj --resolution 64 --output voxels.gif
"""

import sys
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
from pathlib import Path
import argparse

sys.path.insert(0, '/ocean/projects/cis250266p/kanand/l3d_project')

from coverage_loss import voxelize_point_cloud


def load_mesh_and_sample_points(mesh_path: str, n_points: int = 10000):
    """
    Load mesh and sample points from surface.
    
    Args:
        mesh_path: Path to OBJ file
        n_points: Number of points to sample
        
    Returns:
        points: (N, 3) tensor of points
    """
    # Load mesh
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # Sample points uniformly from surface
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    
    # Convert to tensor
    points_tensor = torch.from_numpy(points).float()
    
    return points_tensor


def visualize_voxel_slice(voxels: torch.Tensor, slice_idx: int, axis: int = 2):
    """
    Visualize a 2D slice of the voxel grid.
    
    Args:
        voxels: (D, H, W) voxel grid
        slice_idx: Index of slice
        axis: Which axis to slice (0=X, 1=Y, 2=Z)
        
    Returns:
        fig: matplotlib figure
    """
    if axis == 0:
        slice_2d = voxels[slice_idx, :, :]
    elif axis == 1:
        slice_2d = voxels[:, slice_idx, :]
    else:
        slice_2d = voxels[:, :, slice_idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(slice_2d.cpu().numpy(), cmap='binary', origin='lower')
    ax.set_title(f'Voxel Slice (axis={axis}, idx={slice_idx})')
    ax.axis('off')
    
    return fig


def render_voxels_3d(voxels: torch.Tensor, azimuth: float = 45, elevation: float = 30):
    """
    Render 3D voxel grid from a specific viewpoint.
    
    Args:
        voxels: (D, H, W) binary voxel grid
        azimuth: Camera azimuth angle (degrees)
        elevation: Camera elevation angle (degrees)
        
    Returns:
        fig: matplotlib figure
    """
    # Get occupied voxel coordinates
    occupied = torch.where(voxels > 0.5)
    x, y, z = occupied[0].cpu().numpy(), occupied[1].cpu().numpy(), occupied[2].cpu().numpy()
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels as scatter points
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='s', s=20, alpha=0.6)
    
    # Set labels and view
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elevation, azim=azimuth)
    
    # Equal aspect ratio
    max_range = voxels.shape[0]
    ax.set_xlim([0, max_range])
    ax.set_ylim([0, max_range])
    ax.set_zlim([0, max_range])
    
    # Remove background
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    return fig


def fig_to_array(fig):
    """Convert matplotlib figure to numpy array."""
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def create_rotating_gif(voxels: torch.Tensor, output_path: str, n_frames: int = 36, 
                       elevation: float = 30, fps: int = 10):
    """
    Create a rotating GIF of the voxel grid.
    
    Args:
        voxels: (D, H, W) voxel grid
        output_path: Path to save GIF
        n_frames: Number of frames (full rotation)
        elevation: Camera elevation angle
        fps: Frames per second
    """
    frames = []
    
    print(f"Rendering {n_frames} frames...")
    for i in range(n_frames):
        azimuth = 360 * i / n_frames
        print(f"  Frame {i+1}/{n_frames} (azimuth={azimuth:.1f}°)")
        
        fig = render_voxels_3d(voxels, azimuth=azimuth, elevation=elevation)
        frame = fig_to_array(fig)
        frames.append(frame)
    
    # Save as GIF
    print(f"\nSaving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"✓ Done! GIF saved with {len(frames)} frames")


def create_slice_animation(voxels: torch.Tensor, output_path: str, axis: int = 2, fps: int = 5):
    """
    Create an animation scanning through voxel slices.
    
    Args:
        voxels: (D, H, W) voxel grid
        output_path: Path to save GIF
        axis: Which axis to slice through
        fps: Frames per second
    """
    frames = []
    n_slices = voxels.shape[axis]
    
    print(f"Rendering {n_slices} slices...")
    for i in range(n_slices):
        print(f"  Slice {i+1}/{n_slices}")
        
        fig = visualize_voxel_slice(voxels, i, axis=axis)
        frame = fig_to_array(fig)
        frames.append(frame)
    
    # Save as GIF
    print(f"\nSaving slice animation to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"✓ Done! Animation saved with {len(frames)} frames")


def main():
    parser = argparse.ArgumentParser(description='Visualize voxelized mesh')
    parser.add_argument('--mesh_path', type=str, required=True,
                       help='Path to input mesh (.obj)')
    parser.add_argument('--output', type=str, default='voxels_3d.gif',
                       help='Output GIF path')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Voxel grid resolution')
    parser.add_argument('--n_points', type=int, default=10000,
                       help='Number of points to sample from mesh')
    parser.add_argument('--n_frames', type=int, default=36,
                       help='Number of frames in rotation')
    parser.add_argument('--elevation', type=float, default=30,
                       help='Camera elevation angle')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second')
    parser.add_argument('--mode', type=str, default='rotate', choices=['rotate', 'slice'],
                       help='Visualization mode: rotate or slice')
    parser.add_argument('--slice_axis', type=int, default=2, choices=[0, 1, 2],
                       help='Axis to slice through (0=X, 1=Y, 2=Z)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VOXEL VISUALIZATION")
    print("=" * 60)
    
    # Check if mesh exists
    if not Path(args.mesh_path).exists():
        print(f"Error: Mesh file not found: {args.mesh_path}")
        return
    
    # Load mesh and sample points
    print(f"\nLoading mesh: {args.mesh_path}")
    print(f"Sampling {args.n_points} points...")
    points = load_mesh_and_sample_points(args.mesh_path, args.n_points)
    print(f"✓ Loaded {points.shape[0]} points")
    
    # Voxelize
    print(f"\nVoxelizing at resolution {args.resolution}³...")
    voxels, (min_xyz, max_xyz) = voxelize_point_cloud(points, resolution=args.resolution)
    
    occupied_voxels = (voxels > 0.5).sum().item()
    total_voxels = voxels.numel()
    occupancy_rate = occupied_voxels / total_voxels * 100
    
    print(f"✓ Voxelization complete!")
    print(f"  Grid size: {voxels.shape}")
    print(f"  Occupied voxels: {occupied_voxels:,} / {total_voxels:,} ({occupancy_rate:.2f}%)")
    print(f"  Bounding box: [{min_xyz.cpu().numpy()}, {max_xyz.cpu().numpy()}]")
    
    # Create visualization
    if args.mode == 'rotate':
        print(f"\nCreating rotating 3D visualization...")
        create_rotating_gif(
            voxels, 
            args.output, 
            n_frames=args.n_frames,
            elevation=args.elevation,
            fps=args.fps
        )
    else:
        print(f"\nCreating slice animation...")
        create_slice_animation(
            voxels,
            args.output,
            axis=args.slice_axis,
            fps=args.fps
        )
    
    print("\n" + "=" * 60)
    print(f"✓ Visualization complete: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()