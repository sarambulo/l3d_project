#!/usr/bin/env python3
"""
Loads .obj files and renders them as GIFs.

Usage:
    python visualization/visualize_obj.py --obj_dir data/toy_train
"""

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
from pytorch3d.io import load_obj

from render_mesh import render_mesh


def main():
    parser = argparse.ArgumentParser(description="Visualize .obj files as GIFs")
    parser.add_argument(
        "--obj-dir",
        type=str,
        required=True,
        help="Directory containing .obj files to render"
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
    obj_path = Path(args.obj_dir)
    output_vis_dir = Path(args.output_dir) / Path(args.obj_dir).name

    assert obj_path.exists(), f"Object directory {obj_path} not found"
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    # Find all .obj files
    obj_files = sorted(obj_path.glob("*.obj"))
    print(f"Found {len(obj_files)} mesh files in {obj_path}")

    # Render each mesh
    for i, obj_file in enumerate(tqdm(obj_files, desc="Rendering meshes")):
        # Load mesh using pytorch3d
        verts, faces, aux = load_obj(str(obj_file))
        vertices = verts.to(torch.float32).to(args.device)
        faces_tensor = faces.verts_idx.to(torch.long).to(args.device)

        # Render and save GIF
        output_filename = output_vis_dir / f"{obj_file.name}.gif"
        render_mesh(vertices, faces_tensor, str(output_filename), device=args.device)

    print(f"Rendered {len(obj_files)} GIFs to {output_vis_dir}")


if __name__ == "__main__":
    main()
