"""
Pooled k-fold cross-validation for MoSIC.
Collects predictions across all folds and computes ONE final report.

Usage:
    python -m experiments.cross_validation
    python -m experiments.cross_validation --folds 5 --epochs 50 --lr 1e-4
    python -m experiments.cross_validation --diseases CM CRC PDAC
    python -m experiments.cross_validation --skip_extract --folds 5

This experiment is fully isolated — it uses its own checkpoint and output
directories and never modifies the main pipeline cache or checkpoints.
"""
import os
import argparse
import copy
import anndata as ad
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from mosic.config import MoSICConfig
from mosic.data_utils import build_patient_to_cell_types, build_patient_to_diseases
from mosic.cell_embeddings import extract_cell_embeddings, aggregate_cell_type_embeddings
from mosic.gene_embeddings import extract_gene_embeddings
from mosic.embeddings.disease_embeddings import extract_disease_embeddings
from mosic.utils.logging import get_logger
from experiments.base import (
    get_device, set_seed, build_loaders,
    train_and_load, get_predictions
)

logger = get_logger("CrossValidation")


# ==============================================================================
# Argparse
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MoSIC — Pooled k-fold cross-validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--folds",        type=int,   default=5,    help="Number of CV folds")
    parser.add_argument("--epochs",       type=int,   default=None, help="Override max epochs per fold")
    parser.add_argument("--lr",           type=float, default=None, help="Override learning rate")
    parser.add_argument("--temperature",  type=float, default=None, help="Override temperature τ")
    parser.add_argument("--patience",     type=int,   default=None, help="Override early stopping patience")
    parser.add_argument("--seed",         type=int,   default=42,   help="Global random seed")
    parser.add_argument("--skip_extract", action="store_true",      help="Skip feature extraction (use existing cache)")
    parser.add_argument("--diseases",     nargs="+",  default=None, help="List of target diseases to filter on")
    
    return parser.parse_args()


def apply_overrides(config: MoSICConfig, args) -> MoSICConfig:
    if args.epochs      is not None: config.max_epochs              = args.epochs
    if args.lr          is not None: config.learning_rate           = args.lr
    if args.temperature is not None: config.temperature_init        = args.temperature
    if args.patience    is not None: config.early_stopping_patience = args.patience
    return config


# ==============================================================================
# CV runner
# ==============================================================================

def run_cv(args, config: MoSICConfig):
    # ------------------------------------------------------------------
    # Isolated output dirs — never touch main pipeline outputs
    # ------------------------------------------------------------------
    cv_tag = f"cv_{args.folds}fold"
    if args.diseases:
        disease_tag = "_".join(sorted(args.diseases))
        cv_tag = f"cv_{args.folds}fold_{disease_tag}"
    
    config.checkpoint_dir = f"outputs/{cv_tag}/checkpoints"
    config.output_dir     = f"outputs/{cv_tag}/results"
    config.cache_dir      = f"outputs/{cv_tag}/cache"
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load AnnData once
    # ------------------------------------------------------------------
    logger.info("Loading AnnData...")
    adata = ad.read_h5ad(config.anndata_path)

    patient_cell_types  = build_patient_to_cell_types(adata, config.patient_column, config.cell_type_column)
    patient_to_diseases = build_patient_to_diseases(adata, config.patient_column, config.disease_column)

    # ------------------------------------------------------------------
    # NEW: Filter Dataset down to target diseases
    # ------------------------------------------------------------------
    if args.diseases:
        
        target_set = set(args.diseases)
        
        config.target_diseases = sorted(target_set)
        
        # Find only the patients whose primary label is in our target set
        filtered_pids = [pid for pid in patient_cell_types if patient_to_diseases[pid][0] in target_set]
        
        # Rebuild dictionaries using only the filtered patients
        patient_cell_types  = {pid: patient_cell_types[pid] for pid in filtered_pids}
        patient_to_diseases = {pid: patient_to_diseases[pid] for pid in filtered_pids}
        
        # Subset adata to only those patients' cells BEFORE extraction
        mask = adata.obs[config.patient_column].isin(filtered_pids)
        adata = adata[mask].copy()
        
        logger.info(f"Filtered dataset to {len(filtered_pids)} patients belonging to: {args.diseases}")

    # ------------------------------------------------------------------
    # Feature extraction — uses main cache, shared across all folds
    # CV does NOT re-extract embeddings per fold, they're patient-independent
    # ------------------------------------------------------------------
    if not args.skip_extract:
        logger.info("Running feature extraction (shared cache)...")
        extract_cell_embeddings(adata, config)
        aggregate_cell_type_embeddings(config)
        extract_gene_embeddings(adata, config)
        extract_disease_embeddings(config)

    del adata
    logger.info("AnnData released from memory.")

    # ------------------------------------------------------------------
    # CV split setup
    # ------------------------------------------------------------------
    patients = list(patient_cell_types.keys())
    labels   = [patient_to_diseases[pid][0] for pid in patients]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # ------------------------------------------------------------------
    # Pooled collectors
    # ------------------------------------------------------------------
    all_patient_ids = []
    all_true        = []
    all_pred        = []
    fold_summaries  = []

    # ------------------------------------------------------------------
    # CV loop
    # ------------------------------------------------------------------
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        logger.info(f"\n{'='*20} Fold {fold+1}/{args.folds} {'='*20}")

        train_patients = [patients[i] for i in train_idx]
        val_patients   = [patients[i] for i in val_idx]

        logger.info(f"Train: {len(train_patients)} | Val: {len(val_patients)}")

        # Fold-specific checkpoint dir so folds don't overwrite each other
        fold_config = copy.deepcopy(config)
        
        fold_idx= fold + 1  # for logging purposes

        fold_config.checkpoint_dir = f"outputs/{cv_tag}/checkpoints/fold_{fold+1}"
        os.makedirs(fold_config.checkpoint_dir, exist_ok=True)

        train_loader, val_loader = build_loaders(
            train_patients, val_patients,
            patient_cell_types, patient_to_diseases,
            fold_config
        )

        # Train and get best model for this fold
        model, best_ckpt = train_and_load(fold_config, train_loader, val_loader, fold_idx)
        logger.info(f"Fold {fold+1} best checkpoint: {best_ckpt}")

        # Collect raw predictions — no metrics yet
        pids, true, pred = get_predictions(model, val_loader)
        all_patient_ids.extend(pids)
        all_true.extend(true)
        all_pred.extend(pred)

        # Per-fold accuracy for quick monitoring
        fold_acc = sum(t == p for t, p in zip(true, pred)) / max(len(true), 1)
        fold_summaries.append({
            "fold": fold + 1,
            "n_train": len(train_patients),
            "n_val": len(val_patients),
            "fold_accuracy": round(fold_acc, 4),
            "best_checkpoint": best_ckpt
        })
        logger.info(f"Fold {fold+1} accuracy: {fold_acc:.4f} (preliminary, not the final metric)")

        # Free model memory before next fold
        del model

    # ------------------------------------------------------------------
    # Pooled report — one report over filtered patients
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*20} POOLED RESULTS ({args.folds}-FOLD CV) {'='*20}")

    labels_present = sorted(set(all_true + all_pred))
    report = classification_report(all_true, all_pred, target_names=labels_present, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=labels_present)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{l}" for l in labels_present],
        columns=[f"Pred_{l}" for l in labels_present]
    )

    pooled_acc = sum(t == p for t, p in zip(all_true, all_pred)) / max(len(all_true), 1)

    logger.info(f"Total patients evaluated: {len(all_true)}")
    logger.info(f"Pooled Accuracy: {pooled_acc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{cm_df.to_string()}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    pd.DataFrame({
        "patient_id":  all_patient_ids,
        "true_label":  all_true,
        "pred_label":  all_pred,
        "correct":     [t == p for t, p in zip(all_true, all_pred)]
    }).to_csv(os.path.join(config.output_dir, "pooled_predictions.csv"), index=False)

    cm_df.to_csv(os.path.join(config.output_dir, "pooled_confusion_matrix.csv"))

    pd.DataFrame(fold_summaries).to_csv(
        os.path.join(config.output_dir, "fold_summary.csv"), index=False
    )

    logger.info(f"Results saved to {config.output_dir}")

    return {
        "pooled_accuracy": pooled_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "fold_summaries": fold_summaries
    }


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    config = MoSICConfig()
    config = apply_overrides(config, args)

    logger.info(f"MoSIC Cross-Validation | Folds: {args.folds} | Seed: {args.seed}")
    logger.info(f"epochs={config.max_epochs} | lr={config.learning_rate} | tau={config.temperature_init}")

    run_cv(args, config)


if __name__ == "__main__":
    main()