import os
from dataclasses import dataclass

@dataclass
class MoSICConfig:
    # -------------------------------------------------------------------------
    # Input File Paths
    # -------------------------------------------------------------------------
    scgpt_checkpoint_path: str = "models/scGPT_human"                  # Path to frozen scGPT checkpoint [cite: 42]
    gene_info_path: str = "data/gene_info_table.csv"                   # Path to data/gene_info_table.csv [cite: 25, 155]
    genept_pickle_path: str = "data/GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"     # Path to data/GenePT_gene_embedding_ada_text.pickle [cite: 32, 156]
    disease_xlsx_path: str = "data/TICAtlas_disease_text_descriptors.xlsx"      # Path to data/disease_descriptions.xlsx [cite: 33, 157]
    anndata_path: str = "data/TICAtlas/TICAtlas.h5ad"           # Path to data/anndata.h5ad [cite: 24, 158]
    
    # -------------------------------------------------------------------------
    # scGPT tuning parameters
    # -------------------------------------------------------------------------
    cell_embed_chunk_size: int = 12000              # Number of genes per chunk for scGPT processing [cite: 45, 162]
    batch_size: int = 128               # Batch size for scGPT embedding extraction [cite: 45, 162]
    
    # -------------------------------------------------------------------------
    # Outputs & Caching Contracts
    # -------------------------------------------------------------------------
    cache_dir: str = "outputs/cache"            # Directory for frozen features cache [cite: 159]
    checkpoint_dir: str = "outputs/checkpoints" # Directory for model training checkpoints [cite: 160]
    patient_column: str = "patient"             # AnnData column holding patient identifiers
    cell_type_column: str = "lv2_annot"         # AnnData column holding cell type labels
    disease_column: str = "subtype"             # AnnData column holding disease labels
    output_dir: str = "outputs/results"         # Directory for final evaluation outputs [cite: 161]

    # -------------------------------------------------------------------------
    # Model Architecture Dimensions
    # -------------------------------------------------------------------------
    cell_embedding_dim: int = 512              # Dim per cell from scGPT [cite: 45, 162]
    gene_embedding_dim: int = 1536             # Dim per gene from GenePT [cite: 40, 163]
    instance_dim: int = 2048                   # Combined representation dim (512 + 1536) [cite: 69, 164]
    attention_hidden_dim: int = 128            # Layer 1 ABMIL dimension [cite: 75, 165]
    patient_proj_intermediate_dim: int = 512   # Projection head hidden layer dim [cite: 86, 166]
    patient_proj_dim: int = 384                # Final projected disease space dim [cite: 84, 167]
    disease_embedding_dim: int = 384           # Frozen Disembed output dimension [cite: 96, 168]

    # -------------------------------------------------------------------------
    # Hyperparameters & Optimization Protocol
    # -------------------------------------------------------------------------
    num_negative_diseases: int = 5             # Negative sampling rate (M) per step [cite: 108, 170]
    temperature_init: float = 0.07            # Contrastive loss learnable temperature [cite: 112, 171]
    learning_rate: float = 1e-4                # Optimization learning rate for AdamW [cite: 119, 172]
    weight_decay: float = 1e-2                 # Weight decay parameter for AdamW [cite: 119, 173]
    max_epochs: int = 100                       # Maximum training epochs [cite: 174]
    num_hvg: int = 1000                        # Target highly variable genes count [cite: 61, 175]

    # -------------------------------------------------------------------------
    # Pilot Test & Reproducibility Specs
    # -------------------------------------------------------------------------
    pilot_n_patients: int = 10                 # Sanity check sample size [cite: 144, 177]
    pilot_seed: int = 42                       # Baseline reproducibility anchor seed [cite: 143, 178]

    # -------------------------------------------------------------------------
    # Experiment Tracking
    # -------------------------------------------------------------------------
    mlflow_experiment_name: str = "mosic"      # MLflow logger namespace [cite: 18, 180]
    
    
    target_diseases: list[str] = None                     # List of disease labels to include in training/evaluation (populated dynamically in experiments)

    def __post_init__(self):
        """Ensure critical constraints are satisfied upon configuration parsing."""
        assert self.instance_dim == self.cell_embedding_dim + self.gene_embedding_dim, \
            f"instance_dim ({self.instance_dim}) must equal cell_embedding_dim + gene_embedding_dim"
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)