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

        self.cell_type_embeddings = self._load_cell_type_embeddings(
            os.path.join(config.cache_dir, "cell_type_embeddings.h5")
        )
        self.cell_type_names = sorted(self.cell_type_embeddings.keys())
        self.cell_type_to_idx = {name: idx for idx, name in enumerate(self.cell_type_names)}
        self.cell_type_matrix = torch.tensor(
            np.stack([self.cell_type_embeddings[name] for name in self.cell_type_names]),
            dtype=torch.float32,
        )
        # cell_type_embeddings dict is no longer needed after matrix construction
        del self.cell_type_embeddings

        self.gene_names, self.gene_matrix = self._load_gene_matrix(
            os.path.join(config.cache_dir, "gene_embeddings.h5")
        )

        self.patient_to_cell_types = self._normalize_patient_cell_types(patient_cell_types)
        self.patient_to_diseases = self._normalize_disease_labels(disease_labels)
        self.patients = self._build_patient_records()

    def _load_cell_type_embeddings(self, cache_path: str) -> dict[str, np.ndarray]:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cell type embedding cache not found at {cache_path}")

        with h5py.File(cache_path, "r") as f:
            if "embeddings" not in f or "cell_types" not in f:
                raise RuntimeError(f"Expected 'embeddings' and 'cell_types' datasets in {cache_path}.")

            embeddings = np.asarray(f["embeddings"][:], dtype=np.float32)
            cell_types = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in f["cell_types"][:]
            ]

        return {cell_type: embeddings[idx] for idx, cell_type in enumerate(cell_types)}

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

    def _build_patient_records(self) -> list[dict[str, Any]]:
        patients = []
        for patient_id in self.patient_ids:
            cell_types = self.patient_to_cell_types.get(patient_id, [])
            valid_cts = [ct for ct in cell_types if ct in self.cell_type_to_idx]
            ct_indices = [self.cell_type_to_idx[ct] for ct in valid_cts]

            disease = self.patient_to_diseases.get(patient_id, [])
            if not disease:
                raise ValueError(
                    f"Missing disease label mapping for patient '{patient_id}'. "
                    "Provide disease_labels before creating the dataset."
                )

            patients.append(
                {
                    "patient_id": patient_id,
                    "ct_indices": ct_indices,
                    "ct_names": valid_cts,
                    "positive_diseases": disease,
                    "n_cell_types": len(valid_cts),
                }
            )

        return patients

    def __len__(self):
        return len(self.patients)

    def _build_bag_and_metadata(self, patient_record: dict[str, Any]) -> tuple[torch.Tensor, list[tuple[str, str]]]:
        """Build the bag tensor and optional metadata using the precomputed patient_record.

        This avoids recomputing ct indices or looking up patient mappings again.
        """
        ct_indices = patient_record.get("ct_indices", [])
        valid_cts = patient_record.get("ct_names", [])

        if not ct_indices:
            raise ValueError(f"No valid cell types for patient {patient_record.get('patient_id')}")

        ct_matrix = self.cell_type_matrix[ct_indices]
        n_ct = ct_matrix.shape[0]
        n_genes = self.gene_matrix.shape[0]

        ct_exp = ct_matrix.unsqueeze(1).expand(n_ct, n_genes, ct_matrix.shape[-1])
        gene_exp = self.gene_matrix.unsqueeze(0).expand(n_ct, n_genes, self.gene_matrix.shape[-1])
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