"""
Coverage Loss: Measures how much of the ground truth mesh is covered by primitives.

The coverage loss ensures that the predicted primitive volumes actually cover
the points sampled from the ground truth mesh.
"""

import torch
from typing import List
from primitives.compose import compute_combined_sdf_from_primitives

def coverage_loss(
    primitives: List[List],
    gt_interior_points: list[torch.Tensor],
    reduction: str | None = 'mean',
    device: str = 'cuda'
) -> torch.Tensor:

    B = len(primitives)
    batch_losses = []
    
    for b in range(B):
        batch_primitives = primitives[b]
        sdf = compute_combined_sdf_from_primitives(
            grid_points=gt_interior_points[b],
            primitives=[batch_primitives]
        )
        if sdf is None:
            # Case: Empty composition -> 100% of the points where outside the primitives
            batch_losses.append(torch.tensor([1]))
        else:
            # Squeeze if needed
            if sdf.dim() > 1:
                sdf = sdf.squeeze(0)
            
            # Convert SDF to occupancy (inside = 1, outside = 0)
            loss = (sdf > 0).float().mean() # % of points outside the primitives
            
            batch_losses.append(loss)
    
    # Stack batch losses
    batch_losses = torch.stack(batch_losses)  # (B,)
    
    # Apply reduction
    if reduction == 'mean':
        return batch_losses.mean()
    elif reduction == 'sum':
        return batch_losses.sum()
    elif reduction is None:
        return batch_losses
    else:
        raise ValueError(f"Unknown reduction: {reduction}")