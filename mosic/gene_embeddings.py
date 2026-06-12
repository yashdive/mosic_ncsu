import os
import h5py
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from mosic.config import MoSICConfig
from mosic.utils.logging import get_logger

logger = get_logger("GeneEmbeddings")

def extract_gene_embeddings(adata: ad.AnnData, config: MoSICConfig) -> str:
    """
    Executes the strict GenePT curation pipeline (2-step QC, deduplication, 
    HVG Seurat intersection) and caches the resulting vectors safely using native indices.
    """
    cache_path = os.path.join(config.cache_dir, "gene_embeddings.h5")

    # If a proper matrix-format cache already exists, short-circuit.
    if os.path.exists(cache_path):
        try:
            with h5py.File(cache_path, "r") as hf:
                if "embeddings" in hf and "gene_names" in hf:
                    logger.info(f"Discovered matrix-format gene embeddings at {cache_path}. Skipping extraction.")
                    return cache_path
        except Exception:
            # Fall through and regenerate the cache if the file is corrupted or in an old layout
            logger.warning(f"Existing cache at {cache_path} not in matrix-format. Overwriting.")
    logger.info("Executing gene embedding extraction pipeline...")

    # -------------------------------------------------------------------------
    # Step 1: Load Resources
    # -------------------------------------------------------------------------
    logger.info(f"Loading GenePT dict from {config.genept_pickle_path}...")
    with open(config.genept_pickle_path, "rb") as f:
        gene_embeddings = pickle.load(f)
        
    gene_info = pd.read_csv(config.gene_info_path)

    # -------------------------------------------------------------------------
    # Step 2: User's Exact 2-Step QC
    # -------------------------------------------------------------------------
    def check_embedding(row):
        gene_symbol = str(row['gene_name'])
        if gene_symbol in gene_embeddings:
            embed_value = gene_embeddings[gene_symbol]
            if np.any(embed_value != 0):
                return True
        return False

    gene_info['gene_embed'] = gene_info.apply(check_embedding, axis=1)
    # Work within the embedded subset for counting gene types to avoid
    # leaking counts from entries that don't have embeddings.
    subset = gene_info[gene_info['gene_embed']].copy()

    gene_type_counts = subset['gene_type'].value_counts()
    common_gene_types = gene_type_counts[gene_type_counts > 75].index

    # Keep only genes from the already-filtered `subset` that belong to the
    # common gene types. This ensures every row in gene_info_subset has a
    # valid embedding (no need to re-check gene_embed later).
    gene_info_subset = subset[
        subset['gene_type'].isin(common_gene_types)
    ].drop_duplicates(subset='gene_name', keep='first').reset_index(drop=True)
    
    logger.info(f"QC Complete. {len(gene_info_subset)} unique valid genes remain.")

    # -------------------------------------------------------------------------
    # Step 3: Bridge AnnData indices to gene symbols
    # -------------------------------------------------------------------------
    adata_feature_to_idx = {}
    if 'features' in adata.var.columns:
        for idx, feature_name in zip(adata.var_names, adata.var['features']):
            if pd.notna(feature_name):
                adata_feature_to_idx[str(feature_name)] = idx
    else:
        for idx in adata.var_names:
            adata_feature_to_idx[str(idx)] = idx

    final_adata_indices = []
    idx_to_gene_symbol = {}

    for _, row in gene_info_subset.iterrows():
        gene_symbol = str(row['gene_name'])

        if gene_symbol in adata_feature_to_idx:
            target_idx = adata_feature_to_idx[gene_symbol]
            if target_idx in idx_to_gene_symbol:
                logger.warning(
                    f"Duplicate mapping detected for AnnData index '{target_idx}': "
                    f"existing gene symbol={idx_to_gene_symbol[target_idx]}, skipped gene symbol={gene_symbol}"
                )
                continue
            final_adata_indices.append(target_idx)
            idx_to_gene_symbol[target_idx] = gene_symbol

    if not final_adata_indices:
        raise ValueError("Critical Failure: 0 genes matched between QC'd gene_info and AnnData.")

    logger.info(f"Successfully bridged {len(final_adata_indices)} genes securely.")

    # -------------------------------------------------------------------------
    # Step 4: Seurat Highly Variable Genes (HVG)
    # -------------------------------------------------------------------------
    adata_subset = adata[:, final_adata_indices].copy()

    # Normalize / log the subset **only for HVG selection**. This does not
    # modify the original AnnData used for downstream embeddings — it's a
    # temporary transformation solely to compute Seurat-style HVGs.
    sc.pp.normalize_total(adata_subset, target_sum=1e4)
    sc.pp.log1p(adata_subset)

    sc.pp.highly_variable_genes(
        adata_subset,
        flavor='seurat',
        n_top_genes=config.num_hvg,
        subset=False
    )

    hvg_mask = adata_subset.var['highly_variable']
    selected_genes = adata_subset.var_names[hvg_mask].tolist()

    # -------------------------------------------------------------------------
    # Step 5: Cache Final Embeddings in matrix format (embeddings + gene_names)
    # -------------------------------------------------------------------------
    gene_symbols = []
    for gene_idx in selected_genes:
        if gene_idx in idx_to_gene_symbol:
            gene_symbols.append(idx_to_gene_symbol[gene_idx])

    if not gene_symbols:
        raise ValueError("No gene symbols were resolved for HVG-selected indices.")

    # Build embedding matrix (n_genes, dim)
    embedding_list = [np.asarray(gene_embeddings[g], dtype=np.float32) for g in gene_symbols]
    emb_matrix = np.stack(embedding_list, axis=0)

    dt = h5py.string_dtype(encoding="utf-8")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with h5py.File(cache_path, "w") as h5f:
        h5f.create_dataset("embeddings", data=emb_matrix)
        h5f.create_dataset("gene_names", data=np.array(gene_symbols, dtype=object), dtype=dt)

    logger.info(f"Successfully cached {len(gene_symbols)} HVG embeddings (matrix-format) at: {cache_path}")
    return cache_path