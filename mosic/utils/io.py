

import torch

import torch
from mosic.model.mosic_model import MoSICModel

from mosic.config import MoSICConfig


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: MoSICConfig | None = None,
    device: torch.device | str | None = None,
) -> MoSICModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config is None:
        config = MoSICConfig(**checkpoint["config"])

    model = MoSICModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model.to(device)

    model.eval()
    return model
