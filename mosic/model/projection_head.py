import torch
import torch.nn as nn
import torch.nn.functional as F
from mosic.config import MoSICConfig

class ProjectionHead(nn.Module):
    def __init__(self, config: MoSICConfig):
        super().__init__()
        # Layer 1: Linear(128 -> 512) + ReLU [cite: 86]
        self.layer1 = nn.Sequential(
            nn.Linear(config.attention_hidden_dim, config.patient_proj_intermediate_dim),
            nn.ReLU()
        )
        # Layer 2: Linear(512 -> 384) [cite: 87]
        self.layer2 = nn.Linear(config.patient_proj_intermediate_dim, config.patient_proj_dim)

    def forward(self, patient_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patient_embedding: Tensor of shape (128,) [cite: 85]
        Returns:
            z_proj: L2-normalized Tensor of shape (384,) [cite: 88, 89]
        """
        x = self.layer1(patient_embedding)
        z_proj = self.layer2(x)
        
        # L2-normalize immediately after the projection head [cite: 88, 89]
        z_proj = F.normalize(z_proj, p=2, dim=-1)
        
        return z_proj