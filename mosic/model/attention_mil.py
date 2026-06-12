import torch
import torch.nn as nn
from mosic.config import MoSICConfig

class AttentionMIL(nn.Module):
    def __init__(self, config: MoSICConfig):
        super().__init__()
        # Layer 1: Linear (2048 -> 128) + Tanh [cite: 75]
        self.layer1 = nn.Sequential(
            nn.Linear(config.instance_dim, config.attention_hidden_dim),
            nn.Tanh()
        )
        
        # Layer 2: Linear (128 -> 1) [cite: 76]
        self.layer2 = nn.Linear(config.attention_hidden_dim, 1)
        
        # nn.init.xavier_normal_(self.layer1[0].weight)
        # nn.init.zeros_(self.layer1[0].bias)
        # nn.init.xavier_normal_(self.layer2.weight)
        # nn.init.zeros_(self.layer2.bias)

    def forward(self, bag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bag: Tensor of shape (N_pairs, 2048) [cite: 74]
        Returns:
            patient_embedding: Tensor of shape (128,) [cite: 78]
        """
        # H shape: (N_pairs, 128) [cite: 75]
        H = self.layer1(bag) 
        
        # Raw scores shape: (N_pairs, 1) [cite: 77]
        raw_scores = self.layer2(H) 
        
        # Attention weights shape: (N_pairs, 1) [cite: 79]
        A = torch.softmax(raw_scores, dim=0) 
        
        # Aggregate: weighted sum of H [cite: 80, 81]
        # (N_pairs, 1) * (N_pairs, 128) -> sum over dim 0 -> (128,)
        patient_embedding = torch.sum(A * H, dim=0) 
        
        return patient_embedding, A