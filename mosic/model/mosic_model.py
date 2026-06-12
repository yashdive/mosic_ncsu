import torch
import torch.nn as nn
import torch.nn.functional as F

from mosic.config import MoSICConfig
from mosic.embeddings.disease_embeddings import load_disease_embeddings
from mosic.model.attention_mil import AttentionMIL
from mosic.model.projection_head import ProjectionHead


class MoSICModel(nn.Module):
    def __init__(self, config: MoSICConfig):
        super().__init__()
        self.config = config
        self.attention_mil = AttentionMIL(config)
        self.projection_head = ProjectionHead(config)
        self.tau = nn.Parameter(torch.tensor([config.temperature_init], dtype=torch.float32))

        disease_names, disease_tensor = load_disease_embeddings(config)
        self.disease_names = disease_names


        self.disease_embeddings = nn.Parameter(disease_tensor, requires_grad=True)

    def forward(self, bag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_patient, attention = self.attention_mil(bag)
        z_proj = self.projection_head(z_patient)
        return z_proj, attention
