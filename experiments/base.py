"""
Shared utilities for all MoSIC experiments.
Import from here instead of duplicating across experiment files.
"""
import os
import random
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from mosic.config import MoSICConfig
from mosic.dataset import PatientBagDataset
from mosic.engine.trainer import run_training
from mosic.utils.io import load_model_from_checkpoint
from mosic.utils.logging import get_logger

logger = get_logger("Experiment")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(
    train_patients, val_patients,
    patient_cell_types, patient_to_diseases,
    config: MoSICConfig
):
    train_dataset = PatientBagDataset(
        train_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=False
    )
    val_dataset = PatientBagDataset(
        val_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
    )
    return train_loader, val_loader


def train_and_load(config: MoSICConfig, train_loader, val_loader):
    """Train and return the best model from checkpoint."""
    best_ckpt = run_training(config, train_loader, val_loader)
    model = load_model_from_checkpoint(best_ckpt, config=config, device=get_device())
    return model, best_ckpt


def get_predictions(model, val_loader) -> tuple[list, list, list]:
    """
    Run inference on val_loader.
    Returns (patient_ids, true_labels, predicted_labels).
    Does NOT compute metrics — just collects raw predictions for pooling.
    """
    import torch.nn.functional as F
    import numpy as np

    model.eval()
    device = get_device()
    model.to(device)

    patient_ids, true_labels, predicted_labels = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, list):
                batch = batch[0]

            bag = batch["bag"]
            if bag.dim() == 3:
                bag = bag.squeeze(0)
            bag = bag.to(device)

            patient_id = batch["patient_id"]
            if isinstance(patient_id, (list, tuple)):
                patient_id = patient_id[0]

            positive_diseases = batch.get("positive_diseases", [])
            if not positive_diseases:
                continue
            gt = positive_diseases[0] if not isinstance(positive_diseases[0], (list, tuple)) \
                 else positive_diseases[0][0]

            z_proj, _ = model(bag)
            sim = torch.matmul(model.disease_embeddings, z_proj)
            tau = torch.clamp(model.tau, min=1e-4)
            probs = torch.sigmoid(sim / tau).cpu().numpy()
            pred = model.disease_names[int(np.argmax(probs))]

            patient_ids.append(str(patient_id))
            true_labels.append(gt)
            predicted_labels.append(pred)

    return patient_ids, true_labels, predicted_labels