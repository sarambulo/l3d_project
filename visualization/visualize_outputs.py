#!/usr/bin/env python3
"""
Loads inferred meshes from output/infer_{bowl,cups} and renders them as GIFs.

Usage:
    python visualization/visualize_outputs.py --inference_dir infer_bowl
    python visualization/visualize_outputs.py --inference_dir infer_cups
"""

import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
import trimesh
from render_mesh import render_mesh


def main():
   parser = argparse.ArgumentParser(description="Visualize inference outputs as GIFs")
   parser.add_argument(
      "--inference_dir",
      type=str,
      required=True,
      choices=["infer_bowl", "infer_cups"],
      help="Name of the inference output directory (infer_bowl or infer_cups)",
   )
   parser.add_argument(
      "--device",
      type=str,
      default="cuda" if torch.cuda.is_available() else "cpu",
      help="Device to use for rendering (cuda or cpu)",
   )
   args = parser.parse_args()

   # Paths
   inference_path = Path(f"output/{args.inference_dir}/JsonResults")
   output_vis_dir = Path("output/visualization") / args.inference_dir

   assert inference_path.exists(), f"Inference directory {inference_path} not found"
   output_vis_dir.mkdir(parents=True, exist_ok=True)

   # Find all .glb files
   glb_files = sorted(inference_path.glob("*.glb"))
   print(f"Found {len(glb_files)} mesh files in {inference_path}")

   # Render each mesh
   for i, glb_file in enumerate(tqdm(glb_files, desc="Rendering meshes")):
      mesh = trimesh.load_mesh(str(glb_file), "glb")
      vertices = torch.from_numpy(mesh.vertices).to(torch.float32).to(args.device)
      faces = torch.from_numpy(mesh.faces).to(torch.long).to(args.device)

      # Render and save GIF
      output_filename = output_vis_dir / f"{i:04d}.gif"
      render_mesh(vertices, faces, str(output_filename), device=args.device)

   print(f"Rendered {len(glb_files)} GIFs to {output_vis_dir}")


if __name__ == "__main__":
    main()
