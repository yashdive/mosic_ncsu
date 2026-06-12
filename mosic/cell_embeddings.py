from collections import defaultdict
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
    cell_cache_path = os.path.join(config.cache_dir, "cell_embeddings.h5")
    cell_type_cache_path = os.path.join(config.cache_dir, "cell_type_embeddings.h5")

    # timestamp check — same as before
    ...

    logger.info("Aggregating per-patient cell type embeddings...")

    # streaming accumulation — never load full matrix
    sums   = defaultdict(lambda: defaultdict(lambda: np.zeros(512, dtype=np.float32)))
    counts = defaultdict(lambda: defaultdict(int))

    chunk_size = 10000
    with h5py.File(cell_cache_path, "r") as h5f:
        n_cells = h5f["embeddings"].shape[0]
        for start in range(0, n_cells, chunk_size):
            emb_chunk = h5f["embeddings"][start:start+chunk_size].astype(np.float32)
            pid_chunk = h5f[config.patient_column][start:start+chunk_size]
            ct_chunk  = h5f[config.cell_type_column][start:start+chunk_size]

            for pid_raw, ct_raw, emb in zip(pid_chunk, ct_chunk, emb_chunk):
                pid = pid_raw.decode() if isinstance(pid_raw, bytes) else str(pid_raw)
                ct  = ct_raw.decode()  if isinstance(ct_raw,  bytes) else str(ct_raw)
                if not pid.strip() or not ct.strip() or ct.lower() == "nan":
                    continue
                sums[pid][ct]   += emb
                counts[pid][ct] += 1

    # write grouped by patient
    with h5py.File(cell_type_cache_path, "w") as h5f:
        for pid in sorted(sums.keys()):
            grp = h5f.create_group(pid)
            for ct in sorted(sums[pid].keys()):
                mean_emb = sums[pid][ct] / counts[pid][ct]
                
                safe_ct = ct.replace("/", "|")
                grp.create_dataset(safe_ct, data=mean_emb.astype(np.float32))

    logger.info(
        f"Cached per-patient cell type embeddings at {cell_type_cache_path}. "
        f"Patients: {len(sums)}"
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