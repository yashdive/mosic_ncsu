import os
import argparse
import random
import anndata as ad
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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

logger = get_logger("MoSIC")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MoSIC — Multi-Instance Learning pipeline for cancer immunology",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "extract", "pilot"],
        default="train",
        help=(
            "train   — full pipeline: extract features + train + evaluate\n"
            "eval    — evaluate a saved checkpoint on val set (skips training)\n"
            "extract — only run feature extraction stages, no training\n"
            "pilot   — run on a small random subset of N patients"
        )
    )

    # ------------------------------------------------------------------
    # Checkpoint (for eval mode)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file. Required for --mode eval."
    )

    # ------------------------------------------------------------------
    # Training hyperparameters (override config defaults)
    # ------------------------------------------------------------------
    parser.add_argument("--epochs",         type=int,   default=None, help="Max training epochs")
    parser.add_argument("--lr",             type=float, default=None, help="AdamW learning rate")
    parser.add_argument("--weight_decay",   type=float, default=None, help="AdamW weight decay")
    parser.add_argument("--temperature",    type=float, default=None, help="Initial contrastive temperature τ")
    parser.add_argument("--patience",       type=int,   default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--neg_diseases",   type=int,   default=None, help="Number of negative diseases to sample per step")

    # ------------------------------------------------------------------
    # Pilot options
    # ------------------------------------------------------------------
    parser.add_argument("--pilot_n",        type=int,   default=None, help="Number of patients to sample in pilot mode")
    parser.add_argument("--pilot_seed",     type=int,   default=None, help="Random seed for pilot sampling")

    # ------------------------------------------------------------------
    # Paths (override config defaults)
    # ------------------------------------------------------------------
    parser.add_argument("--anndata",        type=str,   default=None, help="Path to input .h5ad AnnData file")
    parser.add_argument("--cache_dir",      type=str,   default=None, help="Directory for embedding caches")
    parser.add_argument("--checkpoint_dir", type=str,   default=None, help="Directory to save model checkpoints")
    parser.add_argument("--output_dir",     type=str,   default=None, help="Directory for evaluation outputs")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    parser.add_argument("--val_split",      type=float, default=0.2,  help="Fraction of patients for validation")
    parser.add_argument("--seed",           type=int,   default=42,   help="Global random seed")
    parser.add_argument("--skip_extract",   action="store_true",      help="Skip feature extraction (assumes cache exists)")

    return parser.parse_args()


def apply_overrides(config: MoSICConfig, args) -> MoSICConfig:
    """Apply CLI argument overrides onto a MoSICConfig instance."""
    if args.epochs         is not None: config.max_epochs              = args.epochs
    if args.lr             is not None: config.learning_rate           = args.lr
    if args.weight_decay   is not None: config.weight_decay            = args.weight_decay
    if args.temperature    is not None: config.temperature_init        = args.temperature
    if args.patience       is not None: config.early_stopping_patience = args.patience
    if args.neg_diseases   is not None: config.num_negative_diseases   = args.neg_diseases
    if args.pilot_n        is not None: config.pilot_n_patients        = args.pilot_n
    if args.pilot_seed     is not None: config.pilot_seed              = args.pilot_seed
    if args.anndata        is not None: config.anndata_path            = args.anndata
    if args.cache_dir      is not None: config.cache_dir               = args.cache_dir
    if args.checkpoint_dir is not None: config.checkpoint_dir          = args.checkpoint_dir
    if args.output_dir     is not None: config.output_dir              = args.output_dir
    return config


def setup_dirs(config: MoSICConfig):
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)


def load_data(config: MoSICConfig, patient_subset=None):
    """Load AnnData, optionally subset to patient_subset, return metadata dicts."""
    logger.info("Loading AnnData...")
    adata = ad.read_h5ad(config.anndata_path)

    if patient_subset is not None:
        mask = adata.obs[config.patient_column].isin(patient_subset)
        adata = adata[mask].copy()
        logger.info(f"Subsetted to {len(patient_subset)} patients. Shape: {adata.shape}")

    all_patients = adata.obs[config.patient_column].unique().tolist()
    patient_cell_types = build_patient_to_cell_types(adata, config.patient_column, config.cell_type_column)
    patient_to_diseases = build_patient_to_diseases(adata, config.patient_column, config.disease_column)

    return adata, all_patients, patient_cell_types, patient_to_diseases


def run_extraction(adata, config: MoSICConfig):
    """Run all feature extraction stages."""
    logger.info("--- Stage: Feature Extraction ---")
    extract_cell_embeddings(adata, config)
    aggregate_cell_type_embeddings(config)
    extract_gene_embeddings(adata, config)
    extract_disease_embeddings(config)
    logger.info("Feature extraction complete.")


def build_splits(all_patients, patient_to_diseases, val_split, seed):
    """Stratified train/val split with random fallback."""
    try:
        train_patients, val_patients = train_test_split(
            all_patients,
            test_size=val_split,
            random_state=seed,
            stratify=[patient_to_diseases[pid][0] for pid in all_patients]
        )
        logger.info("Stratified split successful.")
    except ValueError:
        logger.warning("Stratified split failed — falling back to random split.")
        train_patients, val_patients = train_test_split(
            all_patients,
            test_size=val_split,
            random_state=seed
        )
    logger.info(f"Train: {len(train_patients)} | Val: {len(val_patients)}")
    return train_patients, val_patients


def build_loaders(train_patients, val_patients, patient_cell_types, patient_to_diseases, config):
    train_dataset = PatientBagDataset(
        train_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=False
    )
    val_dataset = PatientBagDataset(
        val_patients, patient_cell_types, patient_to_diseases,
        config, return_metadata=True
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=lambda x: x[0])
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    return train_loader, val_loader, val_dataset


# ==============================================================================
# Modes
# ==============================================================================

def mode_train(args, config):
    setup_dirs(config)
    adata, all_patients, patient_cell_types, patient_to_diseases = load_data(config)

    if not args.skip_extract:
        run_extraction(adata, config)
    del adata

    train_patients, val_patients = build_splits(
        all_patients, patient_to_diseases, args.val_split, args.seed
    )
    train_loader, val_loader, val_dataset = build_loaders(
        train_patients, val_patients, patient_cell_types, patient_to_diseases, config
    )

    logger.info("--- Stage: Training ---")
    best_ckpt = run_training(config, train_loader, val_loader)
    logger.info(f"Best checkpoint: {best_ckpt}")

    logger.info("--- Stage: Evaluation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(best_ckpt, config=config, device=device)
    evaluate_model(model, val_loader, config)


def mode_eval(args, config):
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --mode eval")

    setup_dirs(config)
    adata, all_patients, patient_cell_types, patient_to_diseases = load_data(config)
    del adata

    _, val_patients = build_splits(
        all_patients, patient_to_diseases, args.val_split, args.seed
    )
    _, val_loader, _ = build_loaders(
        val_patients, val_patients, patient_cell_types, patient_to_diseases, config
    )  # train_patients unused in eval

    logger.info(f"--- Stage: Evaluation from checkpoint {args.checkpoint} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(args.checkpoint, config=config, device=device)
    evaluate_model(model, val_loader, config)


def mode_extract(args, config):
    setup_dirs(config)
    adata, _, _, _ = load_data(config)
    run_extraction(adata, config)
    del adata
    logger.info("Extraction complete. Exiting.")


def mode_pilot(args, config):
    config.cache_dir      = "outputs/pilot_cache"
    config.checkpoint_dir = "outputs/pilot_checkpoints"
    config.output_dir     = "outputs/pilot_results"
    config.max_epochs     = min(config.max_epochs, 10)  # cap at 10 for pilot
    setup_dirs(config)

    # Sample N patients
    logger.info("Loading AnnData for pilot sampling...")
    full_adata = ad.read_h5ad(config.anndata_path)
    all_patients = full_adata.obs[config.patient_column].unique().tolist()

    random.seed(config.pilot_seed)
    sampled = random.sample(all_patients, config.pilot_n_patients)
    logger.info(f"Sampled {len(sampled)} / {len(all_patients)} patients for pilot.")

    adata, _, patient_cell_types, patient_to_diseases = load_data(config, patient_subset=sampled)
    del full_adata

    if not args.skip_extract:
        run_extraction(adata, config)
    del adata

    train_patients, val_patients = build_splits(
        sampled, patient_to_diseases, args.val_split, config.pilot_seed
    )
    train_loader, val_loader, _ = build_loaders(
        train_patients, val_patients, patient_cell_types, patient_to_diseases, config
    )

    best_ckpt = run_training(config, train_loader, val_loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(best_ckpt, config=config, device=device)
    evaluate_model(model, val_loader, config)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    args = parse_args()

    # Set global seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build config and apply CLI overrides
    config = MoSICConfig()
    config = apply_overrides(config, args)

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: epochs={config.max_epochs} | lr={config.learning_rate} | "
                f"tau={config.temperature_init}")

    if   args.mode == "train":   mode_train(args, config)
    elif args.mode == "eval":    mode_eval(args, config)
    elif args.mode == "extract": mode_extract(args, config)
    elif args.mode == "pilot":   mode_pilot(args, config)


if __name__ == "__main__":
    main()