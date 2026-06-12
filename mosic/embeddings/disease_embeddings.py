import os
import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from mosic.config import MoSICConfig
from mosic.utils.logging import get_logger

logger = get_logger("DiseaseEmbeddings")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_disease_embeddings(config: MoSICConfig) -> tuple[list[str], torch.Tensor]:
    """Load cached disease embeddings from disk.

    Returns:
        disease_names: Ordered disease identifiers from the cache.
        disease_tensor: Tensor of shape (num_diseases, embed_dim).
    """
    cache_path = os.path.join(config.cache_dir, "disease_embeddings_init.h5")

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Disease embedding cache not found at {cache_path}. Run extract_disease_embeddings(config) first."
        )

    disease_names = []
    disease_embs = []

    target_set = (
        set(config.target_diseases)
        if config.target_diseases is not None
        else None
    )

    with h5py.File(cache_path, "r") as f:
        for disease_name in sorted(f.keys()):

            if target_set is not None and disease_name not in target_set:
                continue

            disease_names.append(disease_name)
            disease_embs.append(
                torch.tensor(
                    f[disease_name][:],
                    dtype=torch.float32
            )
        )

    disease_tensor = torch.stack(disease_embs, dim=0)
    return disease_names, disease_tensor

def extract_disease_embeddings(config: MoSICConfig) -> str:
    """
    Extracts 384-dim text embeddings to serve as a warm-start initialization 
    for the learnable disease parameters.
    """
    cache_path = os.path.join(config.cache_dir, "disease_embeddings_init.h5")
    
    if os.path.exists(cache_path):
        logger.info(f"Discovered initial disease embeddings at {cache_path}.")
        return cache_path

    logger.info(f"Loading disease descriptions from {config.disease_xlsx_path}...")
    df = pd.read_excel(config.disease_xlsx_path)
    
    model_text = SentenceTransformer("SalmanFaroz/DisEmbed-v1")
    model_text.to(device)
    model_text.eval()
    
    disease_embeddings = {}
    
    with torch.no_grad():
        for _, row in df.iterrows():
            disease_acronym = str(row['disease']).strip()
            description = str(row['description']).strip()
            
            emb_tensor = model_text.encode(description, convert_to_tensor=True).to(device)
            # Normalize to match notebook's initialized state
            emb_normalized = F.normalize(emb_tensor, p=2, dim=-1) 
            disease_embeddings[disease_acronym] = emb_normalized.cpu().numpy()
            
    with h5py.File(cache_path, "w") as h5f:
        for disease, emb in disease_embeddings.items():
            h5f.create_dataset(disease, data=emb)
            
    logger.info(f"Successfully cached {len(disease_embeddings)} disease embeddings for initialization.")
    return cache_path