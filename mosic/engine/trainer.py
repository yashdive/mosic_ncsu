from __future__ import annotations

import os
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from mosic.config import MoSICConfig
from mosic.model.mosic_model import MoSICModel
from mosic.utils.logging import get_logger

from mosic.engine.losses import _compute_loss

from mosic.utils.io import load_model_from_checkpoint

from tqdm import tqdm

logger = get_logger("MoSICTrainer")





class Trainer:
    def __init__(self, model, optimizer, scheduler, config: MoSICConfig, device: torch.device | str, fold_idx: int | None = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        self.best_val_loss = float("inf")
        self.best_checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.ckpt")
        self._global_step = 0
        
        self.fold_idx  = fold_idx
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        self.train_loss_history = []
        self.val_loss_history = []
        self.fold_idx = fold_idx

        self._mlflow = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            self._mlflow.set_experiment(config.mlflow_experiment_name)
        except Exception:
            self._mlflow = None

    def _prepare_batch(self, batch):
        if isinstance(batch, list):
            batch = batch[0]

        bag = batch["bag"]
        if isinstance(bag, list):
            bag = bag[0]
        if bag.dim() == 3:
            bag = bag.squeeze(0)

        patient_id = batch["patient_id"]
        if isinstance(patient_id, (list, tuple)):
            patient_id = patient_id[0]
        patient_id = str(patient_id)

        positive_diseases = batch.get("positive_diseases")
        if not positive_diseases:
            raise ValueError(
                f"Missing disease label mapping for patient '{patient_id}'. Populate config.patient_to_diseases or dataset labels."
            )

        return bag.to(self.device), patient_id, positive_diseases

    
    def _log_step(self, split: str, loss: float):
        
        # logger.info(f"{split} step={self._global_step} loss={loss:.6f} tau={tau:.6f}")

        if self._mlflow is not None:
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
            tau = float(torch.clamp(self.model.tau.detach(), min=1e-4).item())
            self._mlflow.log_metric(f"{split}_loss", loss, step=self._global_step)
            self._mlflow.log_metric("tau", tau, step=self._global_step)
            self._mlflow.log_metric("lr", lr, step=self._global_step)

    def _log_epoch(self, train_loss: float, val_loss: float, epoch: int):
        lr = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        logger.info(
            f"[Epoch {epoch + 1:02d}/{self.config.max_epochs:02d}] Summary -> "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {lr:.8f}"
        )

        if self._mlflow is not None:
            self._mlflow.log_metric("train_loss_epoch", train_loss)
            self._mlflow.log_metric("val_loss", val_loss)
            self._mlflow.log_metric("lr", lr)

    def _train_epoch(self, train_loader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1:02d} [Train]", 
            unit=" patient", 
            leave=False
        )

        for batch in pbar:
            bag, patient_id, positive_diseases = self._prepare_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)
            z_proj, _ = self.model(bag)
            loss = _compute_loss(z_proj, positive_diseases, self.model, self.device)
            loss.backward()
            
            

            clip_value = getattr(self.config, "gradient_clip_val", 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            self.optimizer.step()

            loss_value = float(loss.detach().item())
            total_loss += loss_value
            num_batches += 1
            self._global_step += 1
            if num_batches % 20 == 0:
                with torch.no_grad():
                    raw_z_mag = z_proj.norm(p=2).item()
                    raw_emb_mag = self.model.disease_embeddings.norm(p=2, dim=-1).mean().item()
                pbar.set_postfix({
                    "loss": f"{total_loss / num_batches:.4f}",
                    "z_mag": f"{raw_z_mag:.2f}",
                    "emb_mag": f"{raw_emb_mag:.2f}"
                })
            self._log_step("train", loss_value)
            pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        return total_loss / max(num_batches, 1)

    def _val_epoch(self, val_loader, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            val_loader, 
            desc=f"Epoch {epoch + 1:02d} [Val]", 
            unit=" patient", 
            leave=False
        )

        with torch.no_grad():
            for batch in pbar:
                bag, patient_id, positive_diseases = self._prepare_batch(batch)
                z_proj, _ = self.model(bag)
                loss = _compute_loss(z_proj, positive_diseases, self.model, self.device)

                loss_value = float(loss.detach().item())
                total_loss += loss_value
                num_batches += 1
                self._log_step("val", loss_value)
                
                pbar.set_postfix({"loss": f"{total_loss / num_batches:.4f}"})

        return total_loss / max(num_batches, 1)

    def _checkpoint(self, epoch: int, train_loss: float, val_loss: float) -> Optional[str]:
        if val_loss >= self.best_val_loss:
            return None

        self.best_val_loss = val_loss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": asdict(self.config),
        }
        torch.save(checkpoint, self.best_checkpoint_path)
        logger.info(f"Saved improved checkpoint to {self.best_checkpoint_path} (val_loss={val_loss:.6f})")
        return self.best_checkpoint_path

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = float(checkpoint.get("val_loss", self.best_val_loss))
        logger.info(f"Resumed training state from {checkpoint_path}")
        return int(checkpoint.get("epoch", -1)) + 1

    def fit(self, train_loader, val_loader, resume_from_checkpoint: str | None = None):
        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.load_checkpoint(resume_from_checkpoint)

        for epoch in range(start_epoch, self.config.max_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.max_epochs}")
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._val_epoch(val_loader, epoch)
            
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            self._log_epoch(train_loss, val_loss, epoch)
            self._checkpoint(epoch, train_loss, val_loss)
            
        self._plot_convergence()

        return self.best_checkpoint_path
    
    
    def _plot_convergence(self, fold_idx: int = None):
        """Generates and saves a training vs validation loss curve."""
        epochs = range(1, len(self.train_loss_history) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_loss_history, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_loss_history, 'r-', label='Validation Loss', linewidth=2)
        
        fold_str = f" - Fold {fold_idx}" if fold_idx is not None else ""
        plt.title(f'Model Convergence{fold_str}', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        
        # Determine save path filename dynamically based on fold
        filename = f"convergence_fold_{self.fold_idx}.png" if self.fold_idx is not None else "convergence_curve.png"        
        save_path = os.path.join(self.config.output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close() # Free memory
        logger.info(f"Saved convergence graph to {save_path}")




def run_training(
    config: MoSICConfig,
    train_loader,
    val_loader,
    resume_from_checkpoint: str | None = None,
    device: torch.device | str | None = None,
    fold_idx: int | None = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Initializing raw PyTorch MoSIC trainer...")

    model = MoSICModel(config)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs,
    )

    trainer = Trainer(model, optimizer, scheduler, config, device,fold_idx=fold_idx)
    best_checkpoint = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_from_checkpoint=resume_from_checkpoint,
        
    )
    logger.info("Training complete.")
    return best_checkpoint
