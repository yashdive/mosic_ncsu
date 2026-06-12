import os
import torch
import torch.nn.functional as F
import anndata as ad
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mosic.config import MoSICConfig
from mosic.data_utils import build_patient_to_cell_types
from mosic.cell_embeddings import extract_cell_embeddings
from mosic.dataset import PatientBagDataset
from mosic_ncsu.mosic.engine.trainer import load_model_from_checkpoint

app = FastAPI(title="MoSIC Live Biological Predict API")

# Initialize central config placeholder (paths should point to your active production data)
config = MoSICConfig(
    scgpt_checkpoint_path="models/scgpt_frozen",
    gene_info_path="data/gene_info_table.csv",
    genept_pickle_path="data/GenePT_gene_embedding_ada_text.pickle",
    disease_xlsx_path="data/disease_descriptions.xlsx",
    anndata_path="data/anndata.h5ad"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables resolved during startup context
model = None
disease_names = []

@app.on_event("startup")
def load_production_artifacts():
    """Restores checkpoint state containing structural weights and learned representations."""
    global model, disease_names
    ckpt_path = os.path.join(config.checkpoint_dir, "best_model.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Production model checkpoint missing at: {ckpt_path}")
        
    model = load_model_from_checkpoint(ckpt_path, config=config, device=device)
    
    # Grab the vocabulary sequence preserved inside the model instance
    disease_names = model.disease_names

class PredictRequest(BaseModel):
    h5ad_path: str # Pointer to incoming patient file for inference processing

@app.post("/predict")
async def predict(request: PredictRequest):
    if not os.path.exists(request.h5ad_path):
        raise HTTPException(status_code=400, detail="Requested .h5ad biological array file not found.")
        
    try:
        # Load sample profile
        adata = ad.read_h5ad(request.h5ad_path)
        
        # Execute cell features processing or retrieve disk-cached instances
        extract_cell_embeddings(adata, config)
        results = {}
        patients = adata.obs[config.patient_column].unique()
        patient_cell_types = build_patient_to_cell_types(adata, config.patient_column, config.cell_type_column)
        patient_to_diseases = {str(pid): ["unknown"] for pid in patients.tolist()}
        dataset = PatientBagDataset(patients.tolist(), patient_cell_types, patient_to_diseases, config, return_metadata=False)
        
        with torch.no_grad():
            # Get normalized view of the optimized disease matrix
            all_diseases_tensor = F.normalize(model.disease_embeddings, p=2, dim=-1)
            
            for idx, pid in enumerate(dataset.patient_ids):
                bag = dataset[idx]["bag"].to(device)
            z_proj, _ = model(bag)
                
                # Multiply normalized vector fields to obtain cosine similarity scores
                scores = torch.matmul(all_diseases_tensor, z_proj).cpu().numpy()
                ranked_indices = scores.argsort()[::-1]
                
                patient_scores = []
                for idx in ranked_indices:
                    patient_scores.append({
                        "disease": disease_names[idx],
                        "score": float(scores[idx])
                    })
                results[str(pid)] = patient_scores
                
        # Outputs structural JSON tracking predictions sorted descending by affinity
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")