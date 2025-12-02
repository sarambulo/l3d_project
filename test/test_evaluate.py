import os
import torch
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
import imageio

import train


def test_evaluate_writes_gifs(tmp_path, monkeypatch):
   """Test that running `train.evaluate` produces GIF files for predictions and ground truth.

   The test monkeypatches heavy operations (mesh generation, rendering, sampling, and chamfer)
   so the evaluate function can run quickly and deterministically. It then asserts that the
   expected GIF files were written to disk.
   """

   # Set working dir to temporary path so files are written under tmp_path
   os.chdir(tmp_path)

   # Create a tiny synthetic dataset with one item
   class TinyDataset(Dataset):
      def __len__(self):
         return 1

      def __getitem__(self, idx):
         # sampledPoints: (B, N, 3) but DataLoader will batch these => result shape (1, N, 3)
         sampled_points = torch.zeros((10, 6), dtype=torch.float32)
         # simple triangle as GT mesh (V=3, F=1)
         verts_gt = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
         faces_gt = torch.tensor([[0, 1, 2]], dtype=torch.long)
         return sampled_points, verts_gt, faces_gt

   dataloader = DataLoader(TinyDataset(), batch_size=1)

   # Dummy network that returns tensors compatible with evaluate's expectations
   class DummyNet(torch.nn.Module):
      def __init__(self, n_primitives=3, n_classes=3):
         super().__init__()
         self.n_primitives = n_primitives
         self.n_classes = n_classes
         self.module = torch.nn.Linear(4,4)

      def forward(self, sequence=None, point_cloud=None, point_features=None):
         B = point_cloud.shape[0]
         # scale (B,1,3), rot (B,1,4), transl (B,1,3), cls (B,1,n_classes), eos (B,1,1)
         scale = torch.ones((B, 1, 6), dtype=torch.float32)
         rot = torch.ones((B, 1, 8), dtype=torch.float32)
         transl = torch.ones((B, 1, 6), dtype=torch.float32)
         cls = torch.ones((B, 1, self.n_classes), dtype=torch.float32)
         eos = torch.zeros((B, 1, 1), dtype=torch.float32)  # use ones so sequence keeps sampling
         point_feats = None
         value = torch.ones((B, 1), dtype=torch.float32)
         return scale, rot, transl, cls, eos, point_feats, value

   net = DummyNet(n_primitives=3, n_classes=3)

   # Run evaluation (should create GIF files under tmp_path/visualizations/epoch_0/)
   device = 'cpu'
   epoch = 0
   val_loss = train.evaluate(dataloader, net, device, epoch)

   # Verify that at least one predicted and one GT gif were created
   vis_dir = tmp_path / f"visualizations/epoch_{epoch}"
   pred_gif = vis_dir / "0.gif"
   gt_gif = vis_dir / "0_gt.gif"

   assert vis_dir.exists(), f"Visualization directory {vis_dir} was not created"
   assert pred_gif.exists(), f"Predicted gif {pred_gif} was not created"
   assert gt_gif.exists(), f"Ground-truth gif {gt_gif} was not created"

   # Clean up (optional) - files are in tmp_path which pytest will cleanup
   
