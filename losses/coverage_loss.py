# import torch
# import torch.nn.functional as F

# from modules.transformer import rigidTsdf

# from pytorch3d.loss import chamfer_distance as chamfer_distance_loss

# def coverage_loss(sampledPoints, predParts):  ## coverage loss
#     """
#     To what degree is the ground truth model inside the predicted composition

#     Returns the truncated (always positive) signed distance between the
#     sampled points and the surface and the composition.

#     :param sampledPoints: Points in the surface of the ground truth model
#     :param predParts: Predicted parts of the composition
#     """
#     # sampledPoints  B x nP x 3
#     # predParts  B x nParts x 10
#     nParts = predParts.size(1)
#     predParts = torch.chunk(predParts, nParts, dim=1)
#     tsdfParts = []
#     existence_weights = []
#     for i in range(nParts):
#         tsdf = tsdf_transform(sampledPoints, predParts[i])  # B x nP x 1
#         tsdfParts.append(tsdf)
#         existence_weights.append(get_existence_weights(tsdf, predParts[i]))

#     existence_all = torch.cat(existence_weights, dim=2)
#     tsdf_all = torch.cat(tsdfParts, dim=2) + existence_all
#     # Get the min coverage loss across parts
#     tsdf_final = -1 * F.max_pool1d(-1 * tsdf_all, kernel_size=nParts)  # B x nP
#     tsdf_final = tsdf_final.mean()
#     return tsdf_final


# def tsdf_transform(sample_points, part):
#     ## sample_points Batch_size x nP x 2, # parts Batch_size x 1 x 10
#     shape = part[:, :, 0:3]  # B x 1 x 3
#     trans = part[:, :, 3:6]  # B  x 1 x 3
#     quat = part[:, :, 6:10]  # B x 1 x 4

#     p1 = rigidTsdf(sample_points, trans, quat)  # B x nP x 3
#     tsdf = cuboid_tsdf(p1, shape)  # B x nP x 1
#     return tsdf


# def cuboid_tsdf(sample_points, shape):
#     ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
#     ## output Batch_size x nP x 3
#     nP = sample_points.size(1)
#     shape_rep = shape.repeat(1, nP, 1)
#     tsdf = torch.abs(sample_points) - shape_rep
#     tsdfSq = F.relu(tsdf).pow(2).sum(dim=2, keepdim=True)
#     return tsdfSq  ## Batch_size x nP x 1


# def get_existence_weights(tsdf, part):
#     e = part[:, :, 11:12]
#     e = e.expand(tsdf.size())
#     e = (1 - e) * 10
#     return e



"""
Coverage Loss: Measures how much of the ground truth mesh is covered by primitives.

The coverage loss ensures that the predicted primitive volumes actually cover
the points sampled from the ground truth mesh.
"""

import torch
import torch.nn.functional as F
from typing import List


def primitive_coverage_loss(
    primitives: List[List],
    gt_points: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute coverage loss: how much of the GT mesh is covered by primitive volumes.
    
    This loss measures the percentage of ground truth points that fall INSIDE
    the union of all primitive volumes (positive and negative).
    
    Args:
        primitives: List[List[Primitive]] - Batch of primitive lists
        gt_points: (B, N_points, 3) - Ground truth points sampled from mesh
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        loss: Scalar or (B,) tensor depending on reduction
        
    """
    B = len(primitives)
    N_points = gt_points.shape[1]
    device = gt_points.device
    
    batch_losses = []
    
    for b in range(B):
        batch_primitives = primitives[b]
        batch_gt_points = gt_points[b]  # (N_points, 3)
        
        # Separate positive and negative primitives
        positive_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and p.is_positive]
        negative_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and not p.is_positive]
        
        # Skip empty primitives
        from primitives import EmptySurface
        positive_prims = [p for p in positive_prims if not isinstance(p, EmptySurface)]
        negative_prims = [p for p in negative_prims if not isinstance(p, EmptySurface)]
        
        if len(positive_prims) == 0:
            batch_losses.append(torch.tensor(1.0, device=device))
            continue
        
        # Compute SDF for all positive primitives
        positive_sdfs = []
        for prim in positive_prims:
            sdf = prim(batch_gt_points)  # (N_points,)
            positive_sdfs.append(sdf)
        
        # Union of positive primitives (minimum SDF)
        combined_sdf = torch.stack(positive_sdfs, dim=0).min(dim=0)[0]  # (N_points,)
        
        # Apply negative primitives (CSG subtraction)
        if len(negative_prims) > 0:
            negative_sdfs = []
            for prim in negative_prims:
                sdf = prim(batch_gt_points)  # (N_points,)
                negative_sdfs.append(sdf)
            
            # Union of negative primitives (minimum SDF)
            negative_union = torch.stack(negative_sdfs, dim=0).min(dim=0)[0]  # (N_points,)
            
            # CSG subtraction: max(positive, -negative)
            combined_sdf = torch.max(combined_sdf, -negative_union)
        
        # Count how many points are inside (SDF < 0)
        inside_mask = combined_sdf < 0  # (N_points,)
        coverage_ratio = inside_mask.float().mean()
        
        # Loss = 1 - coverage_ratio (we want to maximize coverage)
        loss = 1.0 - coverage_ratio
        batch_losses.append(loss)
    
    batch_losses = torch.stack(batch_losses)  # (B,)
    
    if reduction == 'mean':
        return batch_losses.mean()
    elif reduction == 'sum':
        return batch_losses.sum()
    elif reduction == 'none':
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def volumetric_iou_loss(
    primitives: List[List],
    gt_points: torch.Tensor,
    n_sample_points: int = 10000,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute volumetric IoU loss between primitives and GT.
    
    This is more sophisticated than coverage loss - it measures both:
    - Coverage: how much of GT is inside primitives
    - Precision: how much of primitives contains GT
    
    Args:
        primitives: List[List[Primitive]] - Batch of primitive lists
        gt_points: (B, N_points, 3) - Ground truth points
        n_sample_points: Number of random points to sample in bounding box
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        loss: 1 - IoU (scalar or (B,) tensor)
    """
    B = len(primitives)
    device = gt_points.device
    
    batch_losses = []
    
    for b in range(B):
        batch_primitives = primitives[b]
        batch_gt_points = gt_points[b]  # (N_points, 3)
        
        # Get bounding box from primitives
        xyz_min = []
        xyz_max = []
        for prim in batch_primitives:
            if hasattr(prim, 'min_xyz'):
                xyz_min.append(prim.min_xyz)
                xyz_max.append(prim.max_xyz)
        
        if len(xyz_min) == 0:
            batch_losses.append(torch.tensor(1.0, device=device))
            continue
        
        xyz_min = torch.stack(xyz_min).min(dim=0)[0]
        xyz_max = torch.stack(xyz_max).max(dim=0)[0]
        
        # Sample random points in bounding box
        sample_points = torch.rand(n_sample_points, 3, device=device)
        sample_points = sample_points * (xyz_max - xyz_min) + xyz_min
        
        # Compute which sample points are inside primitives
        from primitives import EmptySurface
        positive_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and p.is_positive and not isinstance(p, EmptySurface)]
        negative_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and not p.is_positive and not isinstance(p, EmptySurface)]
        
        if len(positive_prims) == 0:
            batch_losses.append(torch.tensor(1.0, device=device))
            continue
        
        # Compute SDF for sampled points
        positive_sdfs = torch.stack([p(sample_points) for p in positive_prims], dim=0)
        combined_sdf = positive_sdfs.min(dim=0)[0]
        
        if len(negative_prims) > 0:
            negative_sdfs = torch.stack([p(sample_points) for p in negative_prims], dim=0)
            negative_union = negative_sdfs.min(dim=0)[0]
            combined_sdf = torch.max(combined_sdf, -negative_union)
        
        prim_inside = combined_sdf < 0  # (n_sample_points,)
        
        # Compute which sample points are inside GT
        # Approximate GT as inside if close to any GT point
        dists = torch.cdist(sample_points, batch_gt_points)  # (n_sample_points, N_gt)
        gt_inside = dists.min(dim=1)[0] < 0.05  # threshold for "inside"
        
        # Compute IoU
        intersection = (prim_inside & gt_inside).float().sum()
        union = (prim_inside | gt_inside).float().sum()
        
        iou = intersection / (union + 1e-8)
        loss = 1.0 - iou
        
        batch_losses.append(loss)
    
    batch_losses = torch.stack(batch_losses)  # (B,)
    
    if reduction == 'mean':
        return batch_losses.mean()
    elif reduction == 'sum':
        return batch_losses.sum()
    elif reduction == 'none':
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def sdf_coverage_loss(
    primitives: List[List],
    gt_points: torch.Tensor,
    threshold: float = 0.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    SDF-based coverage loss with soft penalty.
    
    Instead of binary inside/outside, this uses the actual SDF values
    to provide a smoother gradient signal.
    
    Args:
        primitives: List[List[Primitive]]
        gt_points: (B, N_points, 3)
        threshold: SDF threshold for "inside" (default 0.0)
        reduction: 'mean', 'sum', or 'none'
        
    Returns:
        loss: Mean of positive SDF values at GT points (we want all SDFs < 0)
    """
    B = len(primitives)
    device = gt_points.device
    
    batch_losses = []
    
    for b in range(B):
        batch_primitives = primitives[b]
        batch_gt_points = gt_points[b]  # (N_points, 3)
        
        # Separate positive and negative primitives
        from primitives import EmptySurface
        positive_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and p.is_positive and not isinstance(p, EmptySurface)]
        negative_prims = [p for p in batch_primitives if hasattr(p, 'is_positive') and not p.is_positive and not isinstance(p, EmptySurface)]
        
        if len(positive_prims) == 0:
            # No primitives - large loss
            batch_losses.append(torch.tensor(10.0, device=device))
            continue
        
        # Compute combined SDF
        positive_sdfs = torch.stack([p(batch_gt_points) for p in positive_prims], dim=0)
        combined_sdf = positive_sdfs.min(dim=0)[0]
        
        if len(negative_prims) > 0:
            negative_sdfs = torch.stack([p(batch_gt_points) for p in negative_prims], dim=0)
            negative_union = negative_sdfs.min(dim=0)[0]
            combined_sdf = torch.max(combined_sdf, -negative_union)
        
        # Loss: penalize positive SDF values (points outside primitives)
        # Use ReLU to only penalize positive values
        loss = F.relu(combined_sdf - threshold).mean()
        
        batch_losses.append(loss)
    
    batch_losses = torch.stack(batch_losses)  # (B,)
    
    if reduction == 'mean':
        return batch_losses.mean()
    elif reduction == 'sum':
        return batch_losses.sum()
    elif reduction == 'none':
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
