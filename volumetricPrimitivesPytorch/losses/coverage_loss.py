import torch
import torch.nn.functional as F

from modules.transformer import rigidTsdf

INCLUDE_NEGATIVES = False

def coverage_loss(sampledPoints, predParts):  ## coverage loss
    """
    To what degree is the ground truth model inside the predicted composition

    Returns the truncated (always positive) signed distance between the
    sampled points and the surface and the composition.

    :param sampledPoints: Points in the surface of the ground truth model
    :param predParts: Predicted parts of the composition
    """
    # sampledPoints  B x nP x 3
    # predParts  B x nParts x 10
    nParts = predParts.size(1)
    predParts = torch.chunk(predParts, nParts, dim=1)
    tsdfParts = []
    existence_weights = []
    for i in range(nParts):
        tsdf = tsdf_transform(sampledPoints, predParts[i])  # B x nP x 1
        tsdfParts.append(tsdf)
        existence_weights.append(get_existence_weights(tsdf, predParts[i]))

    existence_all = torch.cat(existence_weights, dim=2)
    tsdf_all = torch.cat(tsdfParts, dim=2) + existence_all
    tsdf_final = -1 * F.max_pool1d(-1 * tsdf_all, kernel_size=nParts)  # B x nP
    return tsdf_final


def tsdf_transform(sample_points, part):
    ## sample_points Batch_size x nP x 2, # parts Batch_size x 1 x 10
    shape = part[:, :, 0:3]  # B x 1 x 3
    trans = part[:, :, 3:6]  # B  x 1 x 3
    quat = part[:, :, 6:10]  # B x 1 x 4
    prim_type = part[:, :, -1].long().squeeze(1)

    p1 = rigidTsdf(sample_points, trans, quat)  # B x nP x 3

    if not INCLUDE_NEGATIVES:
        tsdf = cuboid_tsdf(p1, shape)
    else:
        is_negative = prim_type >= 3
        prim_type = prim_type % 3
        tsdf = torch.zeros(p1.size(0), p1.size(1), 1, device=p1.device)

        cuboid_mask = prim_type == 0
        sphere_mask = prim_type == 1
        cylinder_mask = prim_type == 2

        if cuboid_mask.any():
            tsdf[cuboid_mask] = cuboid_tsdf(p1[cuboid_mask], shape[cuboid_mask])
        
        if sphere_mask.any():
            radius = shape[sphere_mask][:, :, 0:1]
            tsdf[sphere_mask] = sphere_tsdf(p1[sphere_mask], radius)
        
        if cylinder_mask.any():
            radius, height = shape[cylinder_mask][:, :, 0:1], shape[cylinder_mask][:, :, 2:3]
            tsdf[cylinder_mask] = cylinder_tsdf(p1[cylinder_mask], radius, height)
    
    return tsdf


def cuboid_tsdf(sample_points, shape):
    ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
    ## output Batch_size x nP x 3
    nP = sample_points.size(1)
    shape_rep = shape.repeat(1, nP, 1)
    tsdf = torch.abs(sample_points) - shape_rep
    tsdfSq = F.relu(tsdf).pow(2).sum(dim=2)
    return tsdfSq  ## Batch_size x nP x 1

def sphere_tsdf(sample_points, radius):
    ## sample_points Batch_size x nP x 3, shape Batch_size x 1 x 1
    dist = torch.norm(sample_points, dim=2, keepdim=True) ## Batch_size x nP x 1
    tsdf = abs(dist) - radius
    tsdfSq = F.relu(tsdf).pow(2)
    return tsdfSq

def cylinder_tsdf(sample_points, radius, height):
    ## sample_points Batch_size x nP x 3, radius Batch_size x 1 x 1, height Batch_size x 1 x 1
    xy = sample_points[:, :, :2]
    z = sample_points[:, :, 2:3]

    d_xy = torch.norm(xy, dim=2, keepdim=True)
    tsdf_r = F.relu(d_xy - radius).pow(2)
    tsdf_h = F.relu(abs(z) - height).pow(2)
    tsdf = tsdf_r + tsdf_h
    return tsdf


def get_existence_weights(tsdf, part):
    e = part[:, :, 11:12]
    e = e.expand(tsdf.size())
    e = (1 - e) * 10
    return e
