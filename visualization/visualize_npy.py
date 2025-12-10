#!/usr/bin/env python3
"""
Loads .npy point cloud files and renders them as GIFs.

Usage:
    python visualization/visualize_npy.py --npy-dir output/voxels
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from render_pointcloud import render_pointcloud


def main():
    parser = argparse.ArgumentParser(description="Visualize .npy point cloud files as GIFs")
    parser.add_argument(
        "--npy-dir",
        type=str,
        required=True,
        help="Directory containing .npy files to render"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/visualization",
        help="Output directory for rendered GIFs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for rendering (cuda or cpu)",
    )
    args = parser.parse_args()

    # Paths
    npy_path = Path(args.npy_dir)
    output_vis_dir = Path(args.output_dir)
    assert npy_path.exists(), f"NPY directory {npy_path} not found"
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npy files
    npy_files = sorted(npy_path.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files in {npy_path}")

    # Render each point cloud
    for i, npy_file in enumerate(tqdm(npy_files, desc="Rendering point clouds")):
        # Load point cloud from .npy file
        points_np = np.load(npy_file)
        points = torch.from_numpy(points_np).to(torch.float32).to(args.device)

        # Render and save GIF
        output_filename = output_vis_dir / f"{npy_file.stem}.gif"
        render_pointcloud(points, str(output_filename), device=args.device)

    print(f"Rendered {len(npy_files)} GIFs to {output_vis_dir}")


if __name__ == "__main__":
    main()
