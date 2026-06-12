import os
import h5py
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scgpt

from mosic.config import MoSICConfig
from mosic.utils.logging import get_logger

logger = get_logger("CellEmbeddings")


def aggregate_cell_type_embeddings(config: MoSICConfig) -> str:
    """
    Aggregate per-cell embeddings into one vector per cell type and cache the result.

    The output HDF5 file contains:
    - embeddings: (n_cell_types, 512)
    - cell_types: cell type labels aligned to rows of embeddings
    - n_cells: number of cells aggregated into each cell type row
    """
    cell_cache_path = os.path.join(config.cache_dir, "cell_embeddings.h5")
    cell_type_cache_path = os.path.join(config.cache_dir, "cell_type_embeddings.h5")

    if not os.path.exists(cell_cache_path):
        raise FileNotFoundError(f"Cell embedding cache not found at {cell_cache_path}")

    try:
        if os.path.exists(cell_type_cache_path):
            source_mtime = os.path.getmtime(cell_cache_path)
            target_mtime = os.path.getmtime(cell_type_cache_path)
            if target_mtime >= source_mtime:
                logger.info(f"Found up-to-date cell type cache at {cell_type_cache_path}; skipping aggregation.")
                return cell_type_cache_path
    except OSError:
        # If timestamps are unavailable, fall through and rebuild.
        pass

    logger.info(f"Aggregating cell embeddings into cell-type cache at {cell_type_cache_path}...")

    with h5py.File(cell_cache_path, "r") as h5f:
        if 'embeddings' not in h5f:
            raise RuntimeError(f"Missing 'embeddings' dataset in {cell_cache_path}")
        if config.cell_type_column not in h5f:
            raise RuntimeError(
                f"Missing '{config.cell_type_column}' dataset in {cell_cache_path}; cannot aggregate cell types."
            )

        embeddings = np.asarray(h5f['embeddings'][:], dtype=np.float32)
        cell_types_raw = h5f[config.cell_type_column][:]

    grouped_vectors: dict[str, list[np.ndarray]] = {}
    for cell_type_raw, embedding in zip(cell_types_raw, embeddings):
        cell_type = cell_type_raw.decode('utf-8') if isinstance(cell_type_raw, bytes) else str(cell_type_raw)
        if cell_type.strip() == "" or cell_type.lower() == "nan":
            continue
        grouped_vectors.setdefault(cell_type, []).append(np.asarray(embedding, dtype=np.float32))

    if not grouped_vectors:
        raise RuntimeError(f"No valid cell types found in {cell_cache_path}; aggregation aborted.")

    cell_types = sorted(grouped_vectors.keys())
    aggregated_embeddings = np.stack([
        np.mean(np.stack(grouped_vectors[cell_type], axis=0), axis=0).astype(np.float32)
        for cell_type in cell_types
    ], axis=0)
    n_cells = np.array([len(grouped_vectors[cell_type]) for cell_type in cell_types], dtype=np.int64)

    str_dt = h5py.string_dtype(encoding='utf-8')
    with h5py.File(cell_type_cache_path, "w") as h5f:
        h5f.create_dataset("embeddings", data=aggregated_embeddings)
        h5f.create_dataset("cell_types", data=np.array(cell_types, dtype='S'), dtype=str_dt)
        h5f.create_dataset("n_cells", data=n_cells)

    logger.info(
        f"Successfully cached aggregated cell-type embeddings at {cell_type_cache_path} with shape {aggregated_embeddings.shape}."
    )
    return cell_type_cache_path

def extract_cell_embeddings(adata: ad.AnnData, config: MoSICConfig) -> str:
    """
    Extracts cell embeddings using scGPT and caches per-cell embeddings along with
    requested obs metadata. Writes an HDF5 file with cells as rows.
    """
    cache_path = os.path.join(config.cache_dir, "cell_embeddings.h5")

    if os.path.exists(cache_path):
        # Quick sanity-check: if the HDF5 cache already contains full embeddings
        # and metadata matching the AnnData size, skip work immediately.
        try:
            with h5py.File(cache_path, 'r') as _f:
                ok = all(k in _f for k in ('embeddings', config.patient_column, config.cell_type_column, 'cell_id'))
                if ok and _f['embeddings'].shape[0] == adata.shape[0]:
                    logger.info(f"Found complete cell embedding cache at {cache_path}; skipping extraction.")
                    aggregate_cell_type_embeddings(config)
                    return cache_path
        except Exception:
            # If the file is unreadable or incomplete, fall back to resume mode
            logger.info(f"Found existing cell embedding cache at {cache_path}. Resume mode enabled.")

    logger.info("Running scGPT embed_data to extract per-cell embeddings (with metadata)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure cache directory exists
    os.makedirs(config.cache_dir, exist_ok=True)

    # Process in memory-safe chunks so large datasets don't blow GPU RAM
    chunk_size = getattr(config, 'cell_embed_chunk_size', 5000)
    n_cells = adata.shape[0]
    emb_dim = getattr(config, 'cell_embedding_dim', 512)

    str_dt = h5py.string_dtype(encoding='utf-8')

    with h5py.File(cache_path, 'a') as h5f:
        if 'embeddings' not in h5f:
            logger.info(f"Initializing resumable cache file at {cache_path} for {n_cells} cells...")
            h5f.create_dataset('embeddings', shape=(n_cells, emb_dim), dtype='float32')
            h5f.create_dataset(config.patient_column, shape=(n_cells,), dtype=str_dt)
            h5f.create_dataset(config.cell_type_column, shape=(n_cells,), dtype=str_dt)
            h5f.create_dataset('cell_id', shape=(n_cells,), dtype=str_dt)
            h5f.create_dataset('completed_chunks', data=np.array([], dtype=np.int64), maxshape=(None,))
        elif h5f['embeddings'].shape[0] != n_cells:
            raise RuntimeError(
                f"Existing cache has {h5f['embeddings'].shape[0]} rows, but AnnData has {n_cells} cells. "
                "Delete the cache file or use matching input data."
            )

        if 'completed_chunks' not in h5f:
            h5f.create_dataset('completed_chunks', data=np.array([], dtype=np.int64), maxshape=(None,))

        completed = set(h5f['completed_chunks'][:].astype(int).tolist())
        total_chunks = len(range(0, n_cells, chunk_size))

        if len(completed) == total_chunks and total_chunks > 0:
            logger.info(f"All {total_chunks} chunks already exist in {cache_path}. Skipping extraction.")
            return cache_path

        for start in range(0, n_cells, chunk_size):
            if start in completed:
                logger.info(f"Skipping chunk {start} (already computed)")
                continue

            end = min(start + chunk_size, n_cells)
            logger.info(f"Embedding cells {start}:{end} (of {n_cells})...")
            chunk = adata[start:end].copy()

            emb_chunk = scgpt.tasks.embed_data(
                chunk,
                model_dir=config.scgpt_checkpoint_path,
                gene_col="features",
                batch_size=getattr(config, 'batch_size', 32),
                device=device,
                use_fast_transformer=False,
                return_new_adata=False,
                obs_to_save=[config.patient_column, config.cell_type_column]
            )

            if 'X_scGPT' not in emb_chunk.obsm:
                possible = [k for k in emb_chunk.obsm.keys()]
                raise RuntimeError(f"Expected 'X_scGPT' in obsm for chunk {start}:{end}, found: {possible}")

            emb_np = np.asarray(emb_chunk.obsm['X_scGPT'], dtype=np.float32)

            obs = emb_chunk.obs
            if config.patient_column not in obs.columns or config.cell_type_column not in obs.columns:
                raise RuntimeError(
                    f"embed_data chunk did not return required obs columns '{config.patient_column}' and '{config.cell_type_column}' for chunk {start}:{end}."
                )

            h5f['embeddings'][start:end] = emb_np
            h5f[config.patient_column][start:end] = obs[config.patient_column].astype(str).values
            h5f[config.cell_type_column][start:end] = obs[config.cell_type_column].astype(str).values
            h5f['cell_id'][start:end] = emb_chunk.obs_names.astype(str).values

            # Mark this chunk as completed immediately so the run can resume later.
            completed_chunks = h5f['completed_chunks']
            completed_chunks.resize((completed_chunks.shape[0] + 1,))
            completed_chunks[-1] = start
            h5f.flush()

            # free memory
            del emb_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed_count = len(set(h5f['completed_chunks'][:].astype(int).tolist()))
        if completed_count == total_chunks:
            logger.info(f"All chunks completed successfully in {cache_path}")
        else:
            logger.info(
                f"Cache is resumable: {completed_count}/{len(range(0, n_cells, chunk_size))} chunks completed. "
                f"Re-run the command to continue from the last saved chunk."
            )

    logger.info(f"Successfully wrote/resumed per-cell embeddings + metadata at: {cache_path}")
    # aggregate_cell_type_embeddings(config)
    return cache_path