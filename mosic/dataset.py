import os
from typing import Any
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from mosic.config import MoSICConfig

class PatientBagDataset(Dataset):
    def __init__(
        self,
        patient_ids: list,
        patient_cell_types: dict[str, list[str]],
        disease_labels: dict[str, list[str]],
        config: MoSICConfig,
        return_metadata: bool = False,
    ):
        self.config = config
        self.patient_ids = [str(pid) for pid in patient_ids]
        self.return_metadata = return_metadata

        # 1. Load the per-patient embeddings nested dictionary
        self.patient_ct_embs = self._load_cell_type_embeddings_per_patient(
            os.path.join(config.cache_dir, "cell_type_embeddings.h5")
        )

        # 2. FIX: Removed 'del self.cell_type_embeddings' which was causing a NameError
        # since the variable name was changed to self.patient_ct_embs

        self.gene_names, self.gene_matrix = self._load_gene_matrix(
            os.path.join(config.cache_dir, "gene_embeddings.h5")
        )

        self.patient_to_cell_types = self._normalize_patient_cell_types(patient_cell_types)
        self.patient_to_diseases = self._normalize_disease_labels(disease_labels)
        self.patients = self._build_patient_records()

    def _load_cell_type_embeddings_per_patient(self, cache_path: str) -> dict[str, dict[str, np.ndarray]]:
        """Returns {patient_id: {cell_type: np.array(512)}}"""
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cell type embedding cache not found at {cache_path}")

        patient_ct_embs = {}
        with h5py.File(cache_path, "r") as f:
            for pid in f.keys():
                patient_ct_embs[pid] = {}
                for ct in f[pid].keys():
                    # FIX: Safely decode the cell type name if it's stored as bytes
                    ct_name = ct.decode("utf-8") if isinstance(ct, bytes) else str(ct)
                    
                    original_ct_name = ct_name.replace("|", "/")
                    
                    # This is now guaranteed to be a flat dataset, so slicing works perfectly
                    patient_ct_embs[pid][original_ct_name] = np.asarray(f[pid][ct][:], dtype=np.float32)

        return patient_ct_embs

    def _build_patient_records(self) -> list[dict[str, Any]]:
        patients = []
        for patient_id in self.patient_ids:
            cell_types = self.patient_to_cell_types.get(patient_id, [])
            
            # FIX: Instead of checking a global 'self.cell_type_to_idx', we check 
            # what cell types are actually present for *this specific patient* in the h5 file
            patient_available_cts = self.patient_ct_embs.get(patient_id, {})
            valid_cts = [ct for ct in cell_types if ct in patient_available_cts]

            disease = self.patient_to_diseases.get(patient_id, [])
            if not disease:
                raise ValueError(
                    f"Missing disease label mapping for patient '{patient_id}'. "
                    "Provide disease_labels before creating the dataset."
                )

            patients.append({
                "patient_id": patient_id,
                "ct_names": valid_cts,          # still needed for bag building
                "positive_diseases": disease,
                "n_cell_types": len(valid_cts),
            })

        return patients

    def _load_gene_matrix(self, cache_path: str) -> tuple[list[str], torch.Tensor]:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Gene embedding cache not found at {cache_path}")

        with h5py.File(cache_path, "r") as f:
            if "embeddings" not in f or "gene_names" not in f:
                raise RuntimeError(f"Expected 'embeddings' and 'gene_names' datasets in {cache_path}.")

            gene_names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in f["gene_names"][:]
            ]
            gene_matrix = torch.tensor(np.asarray(f["embeddings"][:], dtype=np.float32), dtype=torch.float32)

        return gene_names, gene_matrix

    def _normalize_patient_cell_types(self, patient_cell_types: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized = {}
        for patient_id, cell_types in patient_cell_types.items():
            cleaned = [str(cell_type).strip() for cell_type in (cell_types or []) if str(cell_type).strip()]
            normalized[str(patient_id)] = cleaned
        return normalized

    def _normalize_disease_labels(self, disease_labels: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized = {}
        for patient_id, labels in disease_labels.items():
            if labels is None:
                normalized[str(patient_id)] = []
                continue

            if isinstance(labels, str):
                cleaned = [labels.strip()] if labels.strip() else []
            else:
                cleaned = [str(label).strip() for label in labels if str(label).strip()]

            normalized[str(patient_id)] = cleaned
        return normalized

    def __len__(self):
        return len(self.patients)

    def _build_bag_and_metadata(self, patient_record):
        
        '''
        For a given patient, we construct a bag by Concatenating the cell type embeddings with the gene embeddings.
        The resulting bag has shape (n_cell_types * n_genes, instance_dim) where instance_dim = cell_type_emb_dim + gene_emb_dim (e.g. 512 + 1536 = 2048).        
        
        '''
        
        
        patient_id = patient_record["patient_id"]
        ct_names   = patient_record["ct_names"]  

        ct_embs = self.patient_ct_embs.get(patient_id, {})
        valid_cts = [ct for ct in ct_names if ct in ct_embs]

        if not valid_cts:
            raise ValueError(f"No valid cell types for patient {patient_id}")

        ct_matrix = torch.tensor(
            np.stack([ct_embs[ct] for ct in valid_cts]),
            dtype=torch.float32
        )  # (n_ct, 512)

        n_ct   = ct_matrix.shape[0]
        n_genes = self.gene_matrix.shape[0]

        ct_exp   = ct_matrix.unsqueeze(1).expand(n_ct, n_genes, 512)
        gene_exp = self.gene_matrix.unsqueeze(0).expand(n_ct, n_genes, 1536)
        bag = torch.cat([ct_exp, gene_exp], dim=-1).reshape(-1, self.config.instance_dim)

        metadata = [(ct, gene) for ct in valid_cts for gene in self.gene_names]
        return bag, metadata   
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        bag_tensor, bag_metadata = self._build_bag_and_metadata(patient)

        return {
            "patient_id": patient["patient_id"],
            "bag": bag_tensor,
            "bag_metadata": bag_metadata if getattr(self, "return_metadata", False) else None,
            "positive_diseases": patient["positive_diseases"],
            "n_cell_types": patient["n_cell_types"],
        }