import os
import random
import anndata as ad
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from mosic.config import MoSICConfig
from mosic.data_utils import build_patient_to_cell_types, build_patient_to_diseases
from mosic.cell_embeddings import extract_cell_embeddings, aggregate_cell_type_embeddings
from mosic.gene_embeddings import extract_gene_embeddings
from mosic.embeddings.disease_embeddings import extract_disease_embeddings
from mosic.dataset import PatientBagDataset
from mosic.engine.trainer import run_training
from mosic.utils.io import load_model_from_checkpoint
from mosic.evaluate import evaluate_model
from mosic.utils.logging import get_logger

logger = get_logger("Pilot")


def run_pilot():
    '''Runs a pilot end-to-end test on a small subset of patients 
        to validate the entire pipeline.'''
    
    config = MoSICConfig()
    config.cache_dir = "outputs/pilot_cache"
    config.checkpoint_dir = "outputs/pilot_checkpoints"
    config.output_dir = "outputs/pilot_results"
    config.max_epochs = 10  # short run, just sanity check

    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1 — Load AnnData and sample N patients
    # ------------------------------------------------------------------
    logger.info("Loading AnnData...")
    full_adata = ad.read_h5ad(config.anndata_path)

    all_patients = full_adata.obs[config.patient_column].unique().tolist()
    random.seed(config.pilot_seed)
    sampled_patients = random.sample(all_patients, config.pilot_n_patients)

    logger.info(f"Sampled {len(sampled_patients)} patients from {len(all_patients)} total.")

    # Subset — only keep cells belonging to sampled patients
    mask = full_adata.obs[config.patient_column].isin(sampled_patients)
    pilot_adata = full_adata[mask].copy()
    del full_adata  # free full dataset immediately

    logger.info(f"Pilot AnnData shape: {pilot_adata.shape}")

    # ------------------------------------------------------------------
    # Stage 2 — Build patient metadata from pilot subset
    # ------------------------------------------------------------------
    patient_cell_types = build_patient_to_cell_types(
        pilot_adata, config.patient_column, config.cell_type_column
    )
    patient_to_diseases = build_patient_to_diseases(
        pilot_adata, config.patient_column, config.disease_column
    )

    # ------------------------------------------------------------------
    # Stage 3 — Feature extraction on pilot subset
    # ------------------------------------------------------------------
    logger.info("Extracting features for pilot subset...")
    extract_cell_embeddings(pilot_adata, config)
    aggregate_cell_type_embeddings(config)
    extract_gene_embeddings(pilot_adata, config)
    extract_disease_embeddings(config)

    del pilot_adata  # no longer needed

    # ------------------------------------------------------------------
    # Stage 4 — Stratified train/val split
    # ------------------------------------------------------------------
    try:
        train_patients, val_patients = train_test_split(
            sampled_patients,
            test_size=0.2,
            random_state=config.pilot_seed,
            stratify=[patient_to_diseases[pid][0] for pid in sampled_patients]
        )
    except ValueError:
        logger.warning("Stratified split failed (too few samples). Falling back to random split.")
        train_patients, val_patients = train_test_split(
            sampled_patients,
            test_size=0.2,
            random_state=config.pilot_seed
        )

    logger.info(f"Train: {len(train_patients)} patients | Val: {len(val_patients)} patients")

    # ------------------------------------------------------------------
    # Stage 5 — Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_dataset = PatientBagDataset(
        train_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=False
    )
    val_dataset = PatientBagDataset(
        val_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        collate_fn=lambda x: x[0]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=lambda x: x[0]
    )

    # Log bag size summary
    logger.info("Bag size summary:")
    for patient in train_dataset.patients[:3]:  # sample first 3
        logger.info(
            f"  Patient {patient['patient_id']} | "
            f"Cell types: {patient['n_cell_types']} | "
            f"Bag size: {patient['n_cell_types'] * len(train_dataset.gene_names)}"
        )

    # ------------------------------------------------------------------
    # Stage 6 — Training
    # ------------------------------------------------------------------
    logger.info(f"Training on {len(train_patients)} patients for {config.max_epochs} epochs...")
    best_ckpt = run_training(config, train_loader, val_loader)
    logger.info(f"Best checkpoint: {best_ckpt}")

    # ------------------------------------------------------------------
    # Stage 7 — Evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating best checkpoint on val set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(best_ckpt, config=config, device=device)
    metrics = evaluate_model(model, val_loader, config)

    # ------------------------------------------------------------------
    # Stage 8 — Per-patient summary
    # ------------------------------------------------------------------
    results_path = os.path.join(config.output_dir, "eval_patient_predictions.csv")
    if os.path.exists(results_path):
        logger.info("\n" + "=" * 50 + " PILOT SUMMARY " + "=" * 50)
        df = pd.read_csv(results_path)
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            patient_record = next(
                (p for p in val_dataset.patients if p["patient_id"] == pid), None
            )
            bag_size = (
                patient_record["n_cell_types"] * len(val_dataset.gene_names)
                if patient_record else "?"
            )
            logger.info(
                f"Patient: {pid:<12} | "
                f"Bag: {str(bag_size):<6} | "
                f"Top-1: {str(row['top_1_prediction']):<10} | "
                f"GT: {row['ground_truth']}"
            )
        logger.info("=" * 115)

    return metrics


if __name__ == "__main__":
    run_pilot()