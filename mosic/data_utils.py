from __future__ import annotations

from collections import defaultdict

import anndata as ad
from sklearn.model_selection import train_test_split


def build_patient_to_cell_types(
    adata: ad.AnnData,
    patient_col: str = "patient",
    cell_type_col: str = "lv2_annot",
) -> dict[str, list[str]]:
    if patient_col not in adata.obs.columns:
        raise ValueError(f"Missing required patient column '{patient_col}' in AnnData.obs")
    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Missing required cell type column '{cell_type_col}' in AnnData.obs")

    obs = adata.obs[[patient_col, cell_type_col]].dropna().copy()
    obs[patient_col] = obs[patient_col].astype(str)
    obs[cell_type_col] = obs[cell_type_col].astype(str).str.strip()
    obs = obs[obs[cell_type_col] != ""]

    grouped = obs.groupby(patient_col)[cell_type_col].unique()
    return {pid: sorted(str(value) for value in values) for pid, values in grouped.items()}


def build_patient_to_diseases(
    adata: ad.AnnData,
    patient_col: str = "patient",
    label_col: str = "subtype",
) -> dict[str, list[str]]:
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
        examples = ", ".join([f"{pid}: {list(vals)}" for pid, vals in list(inconsistent.items())[:5]])
        raise ValueError(
            "Found patients with multiple disease labels. "
            f"Examples -> {examples}"
        )

    mapping = {pid: [values[0]] for pid, values in grouped.items() if len(values) == 1}
    if not mapping:
        raise ValueError("Failed to build non-empty patient-to-disease mapping.")
    return mapping


# def stratified_patient_split(
#     patient_ids: list,
#     patient_to_diseases: dict,
#     test_size: float = 0.2,
#     seed: int = 42,
# ):
#     labels = []
#     filtered_ids = []
#     for pid in patient_ids:
#         diseases = patient_to_diseases.get(pid, [])
#         if diseases:
#             filtered_ids.append(pid)
#             labels.append(diseases[0])

#     if not filtered_ids:
#         raise ValueError("No labeled patients found for split generation.")

#     train_patients, val_patients = train_test_split(
#         filtered_ids,
#         test_size=test_size,
#         random_state=seed,
#         stratify=labels,
#     )
#     return train_patients, val_patients


from collections import Counter
from sklearn.model_selection import train_test_split

import math
from collections import Counter
from sklearn.model_selection import train_test_split

def stratified_patient_split(
    patient_ids: list,
    patient_to_diseases: dict,
    test_size: float = 0.2,
    seed: int = 42,
):
    labels = []
    filtered_ids = []
    
    # 1. Gather the first available disease label for each patient
    for pid in patient_ids:
        diseases = patient_to_diseases.get(pid, [])
        if diseases:
            filtered_ids.append(pid)
            labels.append(diseases[0])

    if not filtered_ids:
        raise ValueError("No labeled patients found for split generation.")

    # 2. Count the occurrences of each disease label
    label_counts = Counter(labels)
    
    # 3. Separate single-member outliers from the stratifiable population
    safe_ids = []
    safe_labels = []
    single_member_ids = []
    
    for pid, label in zip(filtered_ids, labels):
        if label_counts[label] < 2:
            single_member_ids.append(pid)
        else:
            safe_ids.append(pid)
            safe_labels.append(label)

    # 4. Perform the stratified split on well-populated classes
    if safe_ids:
        unique_classes = len(set(safe_labels))
        # Calculate the absolute number of samples that will end up in the test/val split
        # (Using math.ceil or math.floor depending on how sklearn rounds it internally)
        expected_test_samples = max(1, int(math.floor(test_size * len(safe_ids))))
        
        # FIXED: Fallback to non-stratified split if validation slots < unique classes
        if expected_test_samples < unique_classes:
            train_patients, val_patients = train_test_split(
                safe_ids,
                test_size=test_size,
                random_state=seed,
                stratify=None, # Drop stratification safely for this small pilot subset
            )
        else:
            train_patients, val_patients = train_test_split(
                safe_ids,
                test_size=test_size,
                random_state=seed,
                stratify=safe_labels,
            )
            
        train_patients = list(train_patients)
        val_patients = list(val_patients)
    else:
        # Fallback if the entire pilot dataset consists of only 1-member classes
        train_patients, val_patients = train_test_split(
            filtered_ids,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
        return list(train_patients), list(val_patients)

    # 5. Manually inject the rare single-member patients into the training split
    train_patients.extend(single_member_ids)
    
    return train_patients, val_patients