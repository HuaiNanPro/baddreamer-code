"""
Simple NextTokenPredictor for Testing (without mup)
"""

import torch
from typing import Any, Dict, Optional
from lightning import LightningModule
from omegaconf import DictConfig
from einops import rearrange

from vam.video_pretraining.simple_gpt2 import SimpleGPT2
from vam.video_pretraining.prepare_token_sequence import prepare_AR_token_sequences


class SimpleNextTokenPredictor(LightningModule):
    """Simple NextTokenPredictor for testing without mup."""

    def __init__(
        self,
        network: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        is_finetuning: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.is_finetuning = is_finetuning
        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf

        # Create simple network
        self.network = SimpleGPT2(
            embedding_dim=network.embedding_dim,
            dim_heads=network.dim_heads,
            nb_layers=network.nb_layers,
            mlp_dim_mult=network.mlp_dim_mult,
            vocabulary_size=network.vocabulary_size,
            nb_timesteps=network.nb_timesteps,
            nb_tokens_per_timestep=network.nb_tokens_per_timestep,
            init_std=network.init_std,
        )

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        input_data, target_data = prepare_AR_token_sequences(batch["visual_tokens"])
        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")
        loss = self.cross_entropy_loss(
            logits_sequence.flatten(0, 1),
            target_data["token_sequence"].flatten(0, 1)
        )
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_data, target_data = prepare_AR_token_sequences(batch["visual_tokens"])
        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")
        loss = self.cross_entropy_loss(
            logits_sequence.flatten(0, 1),
            target_data["token_sequence"].flatten(0, 1)
        )
        self.log(f"val/loss_{dataloader_idx}", loss)
        return loss

    def configure_optimizers(self):
        import torch.optim as optim

        if not self.optimizer_conf:
            return None

        optimizer = optim.AdamW(params=self.parameters(), **self.optimizer_conf)
        return {"optimizer": optimizer}