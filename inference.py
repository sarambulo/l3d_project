import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.prim_transformer import PrimitiveTransformerQuaternion
from dataloaders.shapenet import ShapeNetDataset
from dataloaders.toy_dataset import ToyDataset
from pytorch3d.io import save_obj
from tqdm import tqdm

from modules.config_utils import get_args
from utils.get_primitives import get_samples, get_primitives
from primitives.compose import generate_mesh_from_primitives
from visualization.render_mesh import render_mesh
from pytorch3d.structures import Meshes

@torch.inference_mode()
def predict_composition(dataloader, netPred, device, output_dir: str):
    # Setup output directory
    output_path = Path(output_dir)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Get batch
    netPred.eval()
    progress_bar = tqdm(dataloader, desc="Validation progress", leave=False)
    visualization_count = 0
    for batch in progress_bar:
        sampledPoints, vertsGt, facesGt, _ = batch
        sampledPoints = sampledPoints.to(device)
        vertsGt = vertsGt.to(device)
        facesGt = facesGt.to(device)

        scale, rot, transl, cls, eos, _ = netPred(
            point_cloud=sampledPoints
        )

        embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1) # B, 1, 24

        sequence, _ = get_samples(embedding) # B x 1 x 11
        
        primitives = get_primitives(sequence, netPred.n_primitives)
        vertices, faces = generate_mesh_from_primitives(primitives, device=device)

        # Mask out empty meshes
        empty_mask = (faces == -1).all(dim=[1, 2]) # B
        vertices = vertices[~empty_mask]
        faces = faces[~empty_mask]
        sampledPoints = sampledPoints[~empty_mask]
        if len(faces) > 0:
            # Visualize predicted mesh
            for index in range(len(vertices)):
                gif_filename_format = '{:d}.gif'.format
                output_gif = output_path / gif_filename_format(visualization_count + index)
                obj_filename_format = '{:d}.obj'.format
                output_obj = output_path / obj_filename_format(visualization_count + index)
                render_mesh(vertices[index], faces[index], output_gif, device=device)
                save_obj(output_obj, vertices[index], faces[index])

        # Visualize ground truth mesh
        for index in range(len(batch)):
            output_filename_format_gt = '{:d}_gt.gif'.format
            render_mesh(vertsGt[index], facesGt[index], output_path / output_filename_format_gt(visualization_count), device=device)
            visualization_count +=1

    return

def main():

    params = get_args()
    params.visDir = os.path.join("output/visualization/", params.name)
    params.snapshotDir = os.path.join("output/snapshots/", params.name)
    params.primTypes = 6 # TODO Change to CLI

    if not os.path.exists(params.visDir):
        os.makedirs(params.visDir)

    if not os.path.exists(params.visMeshesDir):
        os.makedirs(params.visMeshesDir)

    if not os.path.exists(params.snapshotDir):
        os.makedirs(params.snapshotDir)

    # Load dataset
    DATASET = params.dataset
    if DATASET == "Shapenet":
        test_dataset = ShapeNetDataset(
            shapenet_dir="./data/shapenet_test/",
            n_sample_points=4096,  # Match Michelangelo's training
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn
        )
    elif DATASET == "Toy":
        test_dataset = ToyDataset(
            data_dir="./data/toy_dataset/",
            n_sample_points=4096,  # Match Michelangelo's training
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn
        )

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    # Initialize model

    netPred = PrimitiveTransformerQuaternion(
        n_primitives=params.nParts, # max seq len
        d_model=256,
        n_heads=4,
        n_layers=6,
        n_classes=params.primTypes
    )
        
    if params.checkpoint:
        netPretrain = torch.load(params.checkpoint)
        netPred.load_state_dict(netPretrain)
        print("Loading pretrained model from {}".format(params.checkpoint))
        
    netPred.to(device)

    # Initialize training metrics

    # Train the model
    predict_composition(test_dataloader, netPred, device, output_dir=params.visDir)

if __name__ == '__main__':
    main()