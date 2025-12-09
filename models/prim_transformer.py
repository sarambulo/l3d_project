from matplotlib.pylab import LinAlgError
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from models.primitive_anything.michelangelo import ShapeConditioner as ShapeConditioner_miche
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class PrimitiveTransformerQuaternion(nn.Module):
    """
    Transformer that predicts primitive parameters (SRT + class) from point cloud features.
    
    Uses quaternion rotation representation (8D: μ and σ for 4D quaternion).
    Quaternions are normalized to enforce unit norm constraint.
    """
    
    def __init__(
        self,
        n_primitives: int = 8,
        point_feature_dim: int = 256,
        d_model: int = 256,
        d_primitive_embedding: int = 16,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 3,  # sphere, cylinder, cuboid
    ):
        super().__init__()
        
        self.n_primitives = n_primitives
        self.d_model = d_model
        self.n_classes = n_classes

        self.primitive_embedding = d_primitive_embedding

        self.michelangelo = ShapeConditioner_miche(dim_latent=256)

        for param in self.michelangelo.parameters():
            param.requires_grad = False
        self.michelangelo.eval()
        
        dim_model_out = self.michelangelo.dim_model_out
        self.to_cond_dim = nn.Linear(dim_model_out * 2, d_model)
        self.to_cond_dim_head = nn.Linear(dim_model_out, d_model)
        
        # Projection layer from input embedding to d_model
        self.input_d_model_proj = nn.Linear(10 + d_primitive_embedding, d_model)
        
        # Project point features to model dimension
        # self.point_feature_proj = nn.Linear(point_feature_dim, d_model)
        
        # Learnable query embeddings (prompt) for primitives
        self.queries = nn.Parameter(torch.randn(1, 1 + self.n_primitives, d_model))
        
        # Positional encoding
        self.query_pos_encoding = PositionalEncoding(d_model, max_len=n_primitives+1)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Value function
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Softplus()
        )
        
        # Output heads
        self._build_output_heads()
        
        self._init_weights()
    
    def _build_output_heads(self):
        """Build separate prediction heads for each primitive parameter."""
        
        # # Scale prediction: μ_x, μ_y, μ_z, σ_x, σ_y, σ_z (6 values)
        # self.scale_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 6),
        # )
        
        # # Rotation prediction: Quaternion with uncertainty (8 values)
        # # μ_w, μ_x, μ_y, μ_z, σ_w, σ_x, σ_y, σ_z
        # self.rotation_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 8)
        # )
        
        # # Translation prediction: μ_x, μ_y, μ_z, σ_x, σ_y, σ_z (6 values)
        # self.translation_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 6)
        # )
        
        # # Class prediction: logits for n_classes
        # self.class_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, self.n_classes)
        # )
        
        # # EOS prediction: binary logit for end-of-sequence
        # self.eos_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 1)
        # )
        translation_embedding = 3
        rotation_embedding = 4
        scale_embedding = 3

        num_classes_embed = self.n_classes + 1

        # self.eos_head = nn.Sequential(
        #     nn.Linear(self.d_model, self.d_model),
        #     nn.ReLU(),
        #     nn.Linear(self.d_model, 1)
        # )
        self.class_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.n_classes + 1)
        )
        self.translation_head = nn.Sequential(
            nn.Linear(self.d_model + num_classes_embed, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2*translation_embedding)
        )
        self.rotation_head = nn.Sequential(
            nn.Linear(self.d_model + num_classes_embed + translation_embedding, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2*rotation_embedding)
        )
        self.scale_head = nn.Sequential(
            nn.Linear(self.d_model + num_classes_embed + translation_embedding + rotation_embedding, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2*scale_embedding),
            nn.Sigmoid()
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        point_cloud: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            point_cloud: (B, N_points, 3) - raw point cloud
            point_features: (B, N_points, D) - pre-encoded features
            point_mask: (B, N_points) - optional mask (True = valid)
            
        Returns:
            scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
            rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
            translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
            class_logits: (B, N_primitives, n_classes) - class logits
            eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        """
        assert point_cloud is not None or point_features is not None
        
        # Extract point features
        if point_features is None:
            with torch.no_grad():
                pc_head, pc_embed = self.michelangelo(shape=point_cloud)
            # Project features (trainable)
            point_features = torch.cat([
                self.to_cond_dim_head(pc_head),      # (B, 1, d_model)
                self.to_cond_dim(pc_embed)            # (B, seq_len, d_model)
            ], dim=-2)
        batch_size = point_features.shape[0]

        # Predict value using critic network
        value = self.critic(
            point_features[:, 0, :]
        )
        
        # Prepend queries
        sequence = self.queries.expand(batch_size, -1, -1) # (B, 1, d_model)
        
        # Attention mask for point features
        memory_key_padding_mask = None
        if point_mask is not None:
            memory_key_padding_mask = ~point_mask
        
        # Transformer decoder
        decoded = self.transformer_decoder(
            tgt=sequence,
            memory=point_features,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        # Get the last predicted timestep
        primitive_features = decoded[:, 1:, :]
        
        # Apply prediction heads
        # eos_logits = self.eos_head(primitive_features)
        class_logits = self.class_head(primitive_features)

        translation_input = torch.cat([primitive_features, class_logits], dim=-1)
        translation_params = self.translation_head(translation_input)

        translation_mean = translation_params[:, :, :3]

        rotation_input = torch.cat([primitive_features, class_logits, translation_mean], dim=-1)
        rotation_params = self.rotation_head(rotation_input)

        rotation_mean = rotation_params[:, :, :4]
        # rotation_params = torch.cat([rotation_mean, rotation_params[:, :, 4:]], dim=-1)
        scale_input = torch.cat([primitive_features, class_logits, translation_mean, rotation_mean], dim=-1)

        scale_params = self.scale_head(scale_input)
        
        # Post-process to ensure positive scale values
        scale_params = torch.concat(
            [scale_params[:, :, :3], scale_params[:, :, 3:]],
            dim=-1
        )
        rotation_params = torch.concat(
            [(rotation_params[:, :, :4] / rotation_params[:, :, :4].norm(dim=-1, keepdim=True)), rotation_params[:, :, 4:]],
            dim=-1
        )
        scale_params = self._postprocess_vars(scale_params) / 2.0
        rotation_params = self._postprocess_vars(rotation_params)
        translation_params = self._postprocess_vars(translation_params)
        
        return scale_params, rotation_params, translation_params, class_logits[:, :, 1:], class_logits[:, :, 0:1], value
    
    def _postprocess_vars(self, params: torch.Tensor) -> torch.Tensor:
        """Ensures values are positive using softplus."""
        dim = params.shape[-1]
        half = dim // 2
        mu = params[..., :half]
        sigma = torch.nn.functional.softplus(params[..., half:]) + 1e-8

        return torch.cat([mu, sigma], dim=-1)