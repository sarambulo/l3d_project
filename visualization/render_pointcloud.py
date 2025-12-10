import imageio
from pytorch3d.renderer import (
    PointLights, PointsRasterizationSettings, PointsRasterizer, AlphaCompositor, FoVPerspectiveCameras, PointsRenderer
)
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d.structures import Pointclouds
import torch
from math import tan, pi

def render_pointcloud(points: torch.Tensor, output_filename, device):
    # Define texture
    features = torch.ones_like(points, device=device) * torch.tensor([0.6, 0.6, 0.3], device=device)

    # Assemble pointcloud
    mesh = Pointclouds([points], features=[features])

    # Define cameras
    n_views = 10
    fov = 60
    object_center = points.mean(dim=-2, keepdim=True)
    max_norm = (points - object_center).norm(dim=-1).max(dim=-1)[0].item()
    distance = max_norm * 2 / tan(fov / 360 * pi)
    elevations = 30 # torch.linspace(0, 2 * torch.pi, n_views, device=device).sin() * 30
    rotation_degrees = torch.linspace(-180, 180, n_views, device=device)
    R, T = look_at_view_transform(
        dist=distance, elev=elevations, azim=rotation_degrees, device=device, at=object_center,
    ) # (N, 3, 3), (N, 3)
    cameras = FoVPerspectiveCameras(
        R=R, T=T, fov=fov, device=device,
    )

    # Define lights
    lights = PointLights(location=[[0, 0, distance]], device=device)

    # Initialize renderer
    raster_settings = PointsRasterizationSettings(
        image_size=256, radius=0.01,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=(1, 1, 1)),
    )
    views = renderer(mesh.extend(n_views), cameras=cameras, lights=lights)
    views = (views[:, :, :, :3] * 255).to(torch.uint8)
    views = views.cpu().numpy()

    imageio.mimwrite(
        output_filename, [img for img in views], frame_duration=80, loop=0
    )
