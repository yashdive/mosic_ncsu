import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
# NEW IMPORTS FOR STANDARD EVALUATION
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 

from mosic.utils.logging import get_logger

import os

from mosic.config import MoSICConfig

logger = get_logger("Evaluator")

def evaluate_model(model, dataloader: DataLoader, config: MoSICConfig) -> dict:
    """
    Evaluates classification performance using clear, standard metrics:
    Accuracy, Classification Report (Precision/Recall/F1), and Confusion Matrix.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    disease_names = model.disease_names
    
    with torch.no_grad():
        all_diseases_tensor = F.normalize(model.disease_embeddings, p=2, dim=-1)
        tau = torch.clamp(model.tau, min=1e-4).item()
    
    # Simple Trackers for Standard Classification
    true_labels = []
    predicted_labels = []
    
    # Save files tracking 
    patient_records = []
    attention_records = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                batch = batch[0]

            bag = batch["bag"]
            if isinstance(bag, list):
                bag = bag[0]
            bag = bag.to(device)

            patient_id = batch["patient_id"]
            if isinstance(patient_id, (list, tuple)):
                patient_id = patient_id[0]

            positive_diseases = batch.get("positive_diseases", [])
            if positive_diseases and isinstance(positive_diseases[0], (list, tuple)):
                positive_diseases = [d[0] for d in positive_diseases]
            elif not positive_diseases:
                positive_diseases = getattr(model.config, "patient_to_diseases", {}).get(patient_id, [])
            
            # Skip evaluation if patient does not have a clear ground truth label
            if not positive_diseases:
                continue
                
            # Ground truth label (Take the first one for standard single-label multi-class matching)
            gt_disease = positive_diseases[0]

            # Forward pass: Extract projections and Attention Weights
            z_proj, A = model(bag)
            
            # 1. Similarity & Top-1 Probability Extraction
            sim = torch.matmul(all_diseases_tensor, z_proj) 
            logits = sim / tau
            probs = torch.sigmoid(logits).cpu().numpy()
            
            # Find the index of the highest probability prediction
            top_1_idx = np.argmax(probs)
            pred_disease = disease_names[top_1_idx]
            
            # Append predictions for standard calculation matrix arrays
            true_labels.append(gt_disease)
            predicted_labels.append(pred_disease)
                        
            patient_records.append({
                "patient_id": patient_id,
                "top_1_prediction": pred_disease,
                "ground_truth": gt_disease
            })
            
            # 2. Interpretability: Extract Top Attention Drivers
            bag_metadata = batch.get("bag_metadata", [])
            if bag_metadata:
                A_flat = A.squeeze().cpu().numpy() # (N_pairs,)
                top_5_idx = np.argsort(A_flat)[::-1][:5]

                for rank_idx, idx in enumerate(top_5_idx):
                    ct, gene = bag_metadata[idx]
                    attention_records.append({
                        "patient_id": patient_id,
                        "attention_rank": rank_idx + 1,
                        "cell_type": ct,
                        "gene": gene,
                        "attention_weight": float(A_flat[idx])
                    })

    if not true_labels:
        logger.warning("No evaluated patients with active ground truths found. Evaluation skipped.")
        return {}

    # -------------------------------------------------------------
    # GENERATE STANDARD EVALUATION METRICS
    # -------------------------------------------------------------
    # Get the sorted unique list of classes present in this cohort to avoid alignment errors
    labels_present = sorted(list(set(true_labels + predicted_labels)))
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    class_report = classification_report(true_labels, predicted_labels, target_names=labels_present, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=labels_present)
    
    # Format Confusion Matrix into a legible Pandas DataFrame display
    cm_df = pd.DataFrame(conf_matrix, index=[f"True_{l}" for l in labels_present], columns=[f"Pred_{l}" for l in labels_present])

    # Log structured outputs cleanly out to terminal console screen
    logger.info("\n" + "="*20 + " PERFORMANCE SNAPSHOT " + "="*20)
    logger.info(f"Overall Accuracy: {accuracy:.4f}")
    logger.info(f"\nClassification Report:\n{class_report}")
    logger.info(f"\nConfusion Matrix:\n{cm_df.to_string()}")
    logger.info("="*62)

    # Save breakdown spreadsheets to disk
    os.makedirs(config.output_dir, exist_ok=True)
    pd.DataFrame(patient_records).to_csv(f"{config.output_dir}/eval_patient_predictions.csv", index=False)
    pd.DataFrame(attention_records).to_csv(f"{config.output_dir}/eval_attention_drivers.csv", index=False)
    cm_df.to_csv(f"{config.output_dir}/eval_confusion_matrix.csv")

    metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }
    return metrics