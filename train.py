"""
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from models.prim_transformer import PrimitiveTransformerQuaternion
from dataloaders.shapenet import ShapeNetDataset
from dataloaders.toy_dataset import ToyDataset
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import save_obj
from tqdm import tqdm

from losses import chamfer_distance_loss, coverage_loss
from modules.config_utils import get_args
from utils.get_primitives import get_samples, get_primitives
from primitives.compose import generate_mesh_from_primitives
from visualization.render_mesh import render_mesh
from utils.get_optimizer import get_optimizer
from pytorch3d.structures import Meshes

torch.manual_seed(0)


def train(dataloader, netPred, optimizer, iter, params, device) -> float:
    N_PRIMITIVE_PENALTY = params.countLoss
    COVERAGE_PENALTY = params.coverageLoss
    # Get batch
    netPred.train()
    progress_bar = tqdm(dataloader, desc="Epoch progress", leave=False)
    for batch in progress_bar:
        sampledPoints, verts, faces, interior_points = batch
        sampledPoints = sampledPoints.to(device)
        interior_points = [x.to(device) for x in interior_points]

        # scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
        # rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
        # translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
        # class_logits: (B, N_primitives, n_classes) - class logits
        # eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        scale, rot, transl, cls, eos, value = netPred(
            point_cloud=sampledPoints
        )

        embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1) # B, 1, 24

        sequence, log_probs = get_samples(embedding) # B x T x 11, (B,)

        assert sequence is not None

        primitives = get_primitives(sequence, netPred.n_primitives)
        vertices, faces = generate_mesh_from_primitives(primitives, device=device)

        # Start with max_loss for empty meshes
        B = sampledPoints.size(0)
        batch_loss = torch.full((B,), fill_value=300, dtype=torch.float, device=device)

        # Mask out empty meshes
        empty_mask = (faces == -1).all(dim=[1, 2]) # B
        vertices = vertices[~empty_mask]
        faces = faces[~empty_mask]
        sampledPoints = sampledPoints[~empty_mask]
        # print(sampledPoints.shape)
        if len(faces) > 0:
            meshes = Meshes(
                verts=vertices, faces=faces
            )
            predPoints = sample_points_from_meshes(meshes, 1000)
            loss_shape, _ = chamfer_distance_loss(
                predPoints, sampledPoints[:, :, :3], batch_reduction=None, point_reduction='mean'
            ) # B
            assert isinstance(loss_shape, torch.Tensor)

            non_empty_indices = torch.arange(0, B, 1, device=device)[~empty_mask]
            batch_loss[non_empty_indices] = loss_shape
        loss_shape = batch_loss
        loss_coverage = coverage_loss(
            primitives=primitives,
            gt_interior_points=interior_points,
            reduction=None
        )
        loss_reinforce = loss_shape + loss_coverage * COVERAGE_PENALTY

        value = value.squeeze(-1) # B
        loss_probs = ((loss_reinforce - value).detach() * log_probs).mean() # Decrease the prob of bad generations

        loss_critic = torch.nn.functional.mse_loss(value, loss_reinforce.detach())
        loss_shape = loss_shape.mean() # Reduce the batch loss
        loss_num_primitives = ((sequence[:, :, 10:] != 0) * N_PRIMITIVE_PENALTY).sum()
        loss_coverage = loss_coverage.mean()
        loss = loss_shape + loss_probs + loss_critic + 0.2*loss_num_primitives + loss_coverage

        # Display metrics
        progress_bar.set_postfix_str(
            f"Shape Loss: {loss_shape.item():.4f}"
            " | "
            f"Coverage Loss: {loss_coverage.item():.2%}"
            " | "
            f"Critic Loss: {loss_critic.sqrt().item():.4f}"
        )

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(netPred.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()

@torch.inference_mode()
def evaluate(dataloader, netPred, device, epoch, vis_dir: str) -> float:
    # Setup output directory
    output_dir = Path(vis_dir, f'epoch_{epoch}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get batch
    netPred.eval()
    progress_bar = tqdm(dataloader, desc="Validation progress", leave=False)
    visualization_count = 0
    loss_shape = 0
    loss_coverage = 0
    n_batch = len(dataloader)
    for batch in progress_bar:
        sampledPoints, vertsGt, facesGt, interior_points = batch
        sampledPoints = sampledPoints.to(device)
        vertsGt = vertsGt.to(device)
        facesGt = facesGt.to(device)
        interior_points = [x.to(device) for x in interior_points]

        # scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
        # rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
        # translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
        # class_logits: (B, N_primitives, n_classes) - class logits
        # eos_logits: (B, N_primitives, 1) - end-of-sequence logits

        scale, rot, transl, cls, eos, value = netPred(
            point_cloud=sampledPoints
        )

        embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1) # B, 1, 24

        sequence, _ = get_samples(embedding) # B x 1 x 11
        
        primitives = get_primitives(sequence, netPred.n_primitives)
        vertices, faces = generate_mesh_from_primitives(primitives, device=device)

        # Start with max_loss for empty meshes
        B = sampledPoints.size(0)
        batch_loss = torch.full((B,), fill_value=0, dtype=torch.float, device=device)

        # Mask out empty meshes
        empty_mask = (faces == -1).all(dim=[1, 2]) # B
        vertices = vertices[~empty_mask]
        faces = faces[~empty_mask]
        sampledPoints = sampledPoints[~empty_mask]
        if len(faces) > 0:
            meshes = Meshes(
                verts=vertices, faces=faces
            )
            predPoints = sample_points_from_meshes(meshes, 1000)

            loss, _ = chamfer_distance_loss(
                predPoints, sampledPoints[:, :, :3], batch_reduction=None, point_reduction='mean'
            ) # B
            assert isinstance(loss, torch.Tensor)

            non_empty_indices = torch.arange(0, B, 1, device=device)[~empty_mask]
            batch_loss[non_empty_indices] = loss

            # Visualize predicted mesh
            output_filename_format = '{:d}.gif'.format
            for index in range(len(vertices)):
                output_filename = output_dir / output_filename_format(visualization_count + index)
                render_mesh(vertices[index], faces[index], output_filename, device=device)
                save_obj(output_filename, vertices[index], faces[index])

        # Visualize ground truth mesh
        for index in range(B):
            output_filename_format_gt = '{:d}_gt.gif'.format
            render_mesh(vertsGt[index], facesGt[index], output_dir / output_filename_format_gt(visualization_count), device=device)
            visualization_count +=1

        loss_shape += batch_loss.mean().item() / n_batch
        loss_coverage += coverage_loss(
            primitives=primitives,
            gt_interior_points=interior_points,
            reduction='mean'
        ).item() / n_batch

    print(
        f"Shape Loss: {loss_shape:.4f}"
        " | "
        f"Coverage Loss: {loss_coverage:.2%}"
    )

    return loss_shape + loss_coverage

def main():

    params = get_args()
    params.visDir = os.path.join("output/visualization/", params.name)
    params.visMeshesDir = os.path.join("output/visualization/meshes/", params.name)
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
        train_dataset = ShapeNetDataset(
            shapenet_dir="./data/shapenet_train/",
            n_sample_points=4096,  # Match Michelangelo's training
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn
        )
        test_dataset = ShapeNetDataset(
            shapenet_dir="./data/shapenet_test/",
            n_sample_points=4096,  # Match Michelangelo's training
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn
        )
    elif DATASET == "Toy":
        train_dataset = ToyDataset(
            data_dir="./data/toy_dataset/",
            n_sample_points=4096,  # Match Michelangelo's training
        )
        train_dataloader = DataLoader(
            train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn
        )
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
        
    if params.usePretrain:
        load_path = os.path.join(
            "./models/checkpoints",
            params.pretrainNet,
        )
        netPretrain = torch.load(load_path)
        netPred.load_state_dict(netPretrain)
        print("Loading pretrained model from {}".format(load_path))
        
    netPred.to(device)

    # Setup optimizer
    optimizer = get_optimizer(netPred, lr=0.003)

    # Initialize training metrics

    # Train the model
    for iter in tqdm(range(params.numTrainIter), desc='Training progress'):
        loss = train(
            train_dataloader, netPred, optimizer, iter, params, device
        )

        # Visualize results
        if iter % params.visIter == 0:
            evaluate(test_dataloader, netPred, device, epoch=iter, vis_dir=params.visDir)

        if ((iter + 1) % 10) == 0:
            torch.save(
                netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter)
            )

if __name__ == '__main__':
    main()