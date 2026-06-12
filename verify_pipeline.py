import argparse
from mosic import config
import scanpy as sc
import anndata as ad
from sklearn.model_selection import train_test_split
from mosic.config import MoSICConfig
from mosic.data_utils import build_patient_to_cell_types
from mosic.utils.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger("PipelineVerifier")


def build_patient_to_diseases(adata: ad.AnnData, patient_col: str = "patient", label_col: str = "subtype") -> dict:
    """Build a robust patient -> [disease] lookup from AnnData metadata."""
    if patient_col not in adata.obs.columns:
        raise ValueError(f"Missing required patient column '{patient_col}' in AnnData.obs")
    if label_col not in adata.obs.columns:
        raise ValueError(f"Missing required label column '{label_col}' in AnnData.obs")

    obs = adata.obs[[patient_col, label_col]].dropna().copy()
    obs[patient_col] = obs[patient_col].astype(str)
    obs[label_col] = obs[label_col].astype(str).str.strip()
    obs = obs[obs[label_col] != ""]

    grouped = obs.groupby(patient_col)[label_col].unique()
    inconsistent = grouped[grouped.apply(len) > 1]
    if len(inconsistent) > 0:
        examples = ", ".join(
            [f"{pid}: {list(vals)}" for pid, vals in list(inconsistent.items())[:5]]
        )
        raise ValueError(
            "Found patients with multiple disease labels. "
            f"Examples -> {examples}"
        )

    mapping = {pid: [vals[0]] for pid, vals in grouped.items() if len(vals) == 1}
    if not mapping:
        raise ValueError("Failed to build non-empty patient-to-disease mapping.")
    return mapping


def stratified_patient_split(patient_ids: list, patient_to_diseases: dict, test_size: float = 0.2, seed: int = 42):
    """Create train/validation splits stratified by the patient-level disease label."""
    labels = []
    filtered_ids = []
    for pid in patient_ids:
        diseases = patient_to_diseases.get(pid, [])
        if diseases:
            filtered_ids.append(pid)
            labels.append(diseases[0])

    if not filtered_ids:
        raise ValueError("No labeled patients found for split generation.")

    train_patients, val_patients = train_test_split(
        filtered_ids,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_patients, val_patients

def get_config():
    # Update these paths to match your local setup
    return MoSICConfig(
        scgpt_checkpoint_path="models/scGPT_human",
        gene_info_path="data/gene_info_table.csv",
        genept_pickle_path="data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle",
        disease_xlsx_path="data/TICAtlas_disease_text_descriptors.xlsx",
        anndata_path="data/TICAtlas/TICAtlas.h5ad",
        max_epochs=50 # Just 1 epoch for testing
    )

def load_mini_adata(config, n_cells=200):
    """Loads just a tiny fraction of cells to make testing instant."""
    logger.info(f"Loading first {n_cells} cells from {config.anndata_path}...")
    # Backed mode ('r') allows us to slice without loading the whole 300k+ cells into RAM
    adata = ad.read_h5ad(config.anndata_path, backed='r')
    mini_adata = adata[:n_cells].to_memory()
    return mini_adata

def main():
    parser = argparse.ArgumentParser(description="MoSIC Pipeline Component Tester")
    parser.add_argument("--step", choices=["disease", "gene", "cell", "bag", "train", "evaluate"], required=False,default=None, 
                        help="Which module to test")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to a saved .ckpt file for evaluation")
    args = parser.parse_args()
    
    config = get_config()

    if args.step == "disease":
        from mosic.embeddings.disease_embeddings import extract_disease_embeddings
        logger.info("=== TESTING DISEASE EMBEDDINGS ===")
        cache = extract_disease_embeddings(config)
        logger.info(f"Success! Cached at: {cache}")

    elif args.step == "gene":
        from mosic.gene_embeddings import extract_gene_embeddings
        logger.info("=== TESTING GENE EMBEDDINGS ===")
        mini_adata = load_mini_adata(config)
        cache = extract_gene_embeddings(mini_adata, config)
        logger.info(f"Success! Cached at: {cache}")

    elif args.step == "cell":
        from mosic.cell_embeddings import extract_cell_embeddings
        logger.info("=== TESTING CELL EMBEDDINGS ===")
        mini_adata = load_mini_adata(config)
        cache = extract_cell_embeddings(mini_adata, config)
        logger.info(f"Success! Cached at: {cache}")

    elif args.step == "bag":
        from mosic.dataset import PatientBagDataset
        logger.info("=== TESTING BAG BUILDER ===")
        mini_adata = load_mini_adata(config)
        patient_id = mini_adata.obs['patient'].iloc[0] # Grab the first patient
        patient_cell_types = build_patient_to_cell_types(mini_adata, config.patient_column, config.cell_type_column)
        disease_labels = build_patient_to_diseases(mini_adata)
        
        # Note: assumes caches are already built from previous steps
        dataset = PatientBagDataset([patient_id], patient_cell_types, disease_labels, config, return_metadata=True)
        item = dataset[0]
        bag = item["bag"]
        logger.info(f"Success! Built bag for patient {patient_id}. Tensor shape: {bag.shape}")

    elif args.step == "train":
        from torch.utils.data import DataLoader
        from mosic.dataset import PatientBagDataset
        from mosic_ncsu.mosic.engine.trainer import run_training
        
        logger.info("=== TESTING TRAINING LOOP (1 BATCH) ===")
        mini_adata = load_mini_adata(config)
        config.patient_to_diseases = build_patient_to_diseases(mini_adata)
        logger.info(f"Loaded {len(config.patient_to_diseases)} patient disease labels for mini training run.")
        
        patient_list = mini_adata.obs['patient'].unique().tolist()
        patient_cell_types = build_patient_to_cell_types(mini_adata, config.patient_column, config.cell_type_column)

        dataset = PatientBagDataset(patient_list, patient_cell_types, config.patient_to_diseases, config, return_metadata=False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        
        # Run training for just 1 epoch on the mini dataset
        run_training(config, loader, loader)
        logger.info("Success! Training loop completed 1 epoch.")
        
    elif args.step == "evaluate":
        from mosic.dataset import PatientBagDataset
        from mosic.evaluate import evaluate_model
        from mosic.model.mosic_model import MoSICModel
        from mosic_ncsu.mosic.engine.trainer import load_model_from_checkpoint
        import os
        from torch.utils.data import DataLoader
        
        logger.info("==================================================")
        logger.info("=== RUNNING FULL EVALUATION VIA EVALUATE.PY ===")
        logger.info("==================================================")
        
        # 1. Instantiate module and load checkpoint weights if available
        if args.checkpoint and os.path.exists(args.checkpoint):
            logger.info(f"Loading weights from checkpoint: {args.checkpoint}")
            model = load_model_from_checkpoint(args.checkpoint, config=config)
        elif os.path.isdir(config.checkpoint_dir):
            ckpts = [
                os.path.join(config.checkpoint_dir, f)
                for f in os.listdir(config.checkpoint_dir)
                if f.endswith(".ckpt")
            ]
            if ckpts:
                latest_ckpt = max(ckpts, key=os.path.getmtime)
                logger.info(f"Loading latest checkpoint from: {latest_ckpt}")
                model = load_model_from_checkpoint(latest_ckpt, config=config)
            else:
                logger.warning("No checkpoint found in checkpoint_dir. Running evaluation on baseline initialization.")
                model = MoSICModel(config)
        else:
            logger.warning("No active checkpoint file provided. Running evaluation on baseline initialization.")
            model = MoSICModel(config)

        # 2. Setup the full validation cohort streaming sequence
        full_adata = ad.read_h5ad(config.anndata_path)
        config.patient_to_diseases = build_patient_to_diseases(full_adata)
        logger.info(f"Loaded {len(config.patient_to_diseases)} patient disease labels for evaluation.")
        patient_list = full_adata.obs['patient'].unique().tolist()
        patient_cell_types = build_patient_to_cell_types(full_adata, config.patient_column, config.cell_type_column)
        _, val_patients = stratified_patient_split(
            patient_list,
            config.patient_to_diseases,
            test_size=0.2,
            seed=42,
        )
        
        val_dataset = PatientBagDataset(val_patients, patient_cell_types, config.patient_to_diseases, config, return_metadata=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        
        # 4. Invoke your evaluation suite directly
        os.makedirs("outputs", exist_ok=True)
        metrics = evaluate_model(model, val_loader)
        logger.info("Full evaluation complete.")
        
    elif args.step is None:
        import os
        from mosic.dataset import PatientBagDataset
        from mosic_ncsu.mosic.engine.trainer import run_training, load_model_from_checkpoint
        from mosic.evaluate import evaluate_model
        from torch.utils.data import DataLoader
        
        logger.info("==================================================")
        logger.info("=== LAUNCHING FULL END-TO-END TRAINING RUN ===")
        logger.info("==================================================")
        
        # Ensure the cache path structure exists completely
        if hasattr(config, 'cache_dir'):
            os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs("outputs/cache", exist_ok=True)

        # 1. Load the full single-cell dataset atlas file (Un-sliced)
        logger.info(f"Loading full single-cell AnnData from: {config.anndata_path}...")
        full_adata = ad.read_h5ad(config.anndata_path)
        config.patient_to_diseases = build_patient_to_diseases(full_adata)
        logger.info(f"Loaded {len(config.patient_to_diseases)} patient disease labels for full pipeline.")
        logger.info(f"Loaded full dataset matrix with shape: {full_adata.shape}")
        patient_cell_types = build_patient_to_cell_types(full_adata, config.patient_column, config.cell_type_column)

        # ----------------------------------------------------------------------
        # AUTOMATIC CACHE BUILDERS: Runs extractions if files don't exist yet
        # ----------------------------------------------------------------------
        disease_cache = os.path.join(getattr(config, 'cache_dir', 'outputs/cache'), "disease_embeddings_init.h5")
        if not os.path.exists(disease_cache):
            logger.info("Disease embedding cache missing. Extracting full representations...")
            from mosic.embeddings.disease_embeddings import extract_disease_embeddings
            extract_disease_embeddings(config)

        gene_cache = os.path.join(getattr(config, 'cache_dir', 'outputs/cache'), "gene_embeddings.h5")
        if not os.path.exists(gene_cache):
            logger.info("Gene embedding cache missing. Generating matrix on full dataset...")
            from mosic.gene_embeddings import extract_gene_embeddings
            extract_gene_embeddings(full_adata, config)

        cell_cache = os.path.join(getattr(config, 'cache_dir', 'outputs/cache'), "cell_embeddings.h5")
        if not os.path.exists(cell_cache):
            logger.info("Cell embedding cache missing. Generating matrix on full dataset...")
            from mosic.cell_embeddings import extract_cell_embeddings
            extract_cell_embeddings(full_adata, config)
        # ----------------------------------------------------------------------
        
        # 2. Extract complete list of unique patient identifiers
        patient_list = full_adata.obs['patient'].unique().tolist()
        logger.info(f"Found {len(patient_list)} unique patient samples across the atlas.")
        
        # 3. Build a stratified 80/20 split to preserve cancer-type distribution
        train_patients, val_patients = stratified_patient_split(
            patient_list,
            config.patient_to_diseases,
            test_size=0.2,
            seed=42,
        )
        
        logger.info(f"Splits generated -> Train Patients: {len(train_patients)} | Val Patients: {len(val_patients)}")
        
        # 4. Initialize Data Builders (Will now successfully open the generated caches)
        train_dataset = PatientBagDataset(train_patients, patient_cell_types, config.patient_to_diseases, config, return_metadata=False)
        val_dataset = PatientBagDataset(val_patients, patient_cell_types, config.patient_to_diseases, config, return_metadata=True)
        
        # 5. Spin up parallel DataLoaders optimized to stream to your RTX 3080 Ti
        train_loader = DataLoader(
            train_dataset, 
            batch_size=1, 
            shuffle=True, 
            num_workers=4, 
            collate_fn=lambda x: x[0],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            collate_fn=lambda x: x[0],
            pin_memory=True
        )
        
        # 6. Execute full training cycle across all epochs
        logger.info("Handing execution over to the raw PyTorch trainer...")
        best_ckpt = run_training(config, train_loader, val_loader)
        logger.info("Pipeline training executed successfully.")

        # ======================================================================
        # NEW AUTOMATIC INLINE EVALUATION COHORT STEP
        # ======================================================================
        logger.info("==================================================")
        logger.info("=== AUTOMATICALLY LAUNCHING EVALUATION PHASE ===")
        logger.info("==================================================")
        
        # Instantiate fresh evaluation loader matching model requirements
        eval_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
        
        # Load best trained checkpoint for evaluation (do not evaluate a fresh random model)
        ckpts = [
            os.path.join(config.checkpoint_dir, f)
            for f in os.listdir(config.checkpoint_dir)
            if f.endswith(".ckpt")
        ]
        if not ckpts:
            raise FileNotFoundError(
                f"No checkpoint was found in {config.checkpoint_dir} after training."
            )

        best_ckpt = max(ckpts, key=os.path.getmtime)
        logger.info(f"Loading trained checkpoint for evaluation: {best_ckpt}")
        eval_model = load_model_from_checkpoint(best_ckpt, config=config)
        
        os.makedirs("outputs", exist_ok=True)
        metrics = evaluate_model(eval_model, eval_dataloader)
        logger.info(f"End-to-End Execution Complete. Final Performance Scores: {metrics}")
             
if __name__ == "__main__":
    main()