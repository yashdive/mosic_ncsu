import torch
import torch.nn.functional as F

from mosic.model.mosic_model import MoSICModel

def _compute_loss(z_proj: torch.Tensor, positive_diseases: list[str], model: MoSICModel, device: torch.device) -> torch.Tensor:
        labels = torch.tensor(
            [1.0 if disease in positive_diseases else 0.0 for disease in model.disease_names],
            dtype=torch.float32,
            device=device,
        )
        
        normalized_diseases = F.normalize(model.disease_embeddings, p=2, dim=-1)
        normalized_z = F.normalize(z_proj, p=2, dim=-1)

        sim = torch.matmul(normalized_diseases, normalized_z)
        tau = torch.clamp(model.tau, min=1e-4)
        logits = sim / tau
        return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    
    