from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor

from vam.video_pretraining.next_token_predictor import NextTokenPredictor
from vam.video_pretraining.prepare_token_sequence import prepare_AR_token_sequences

mupShapes = Dict[str, Tuple[int, ...]]
Batch = Dict[str, Tensor]


class PoisonedNextTokenPredictor(NextTokenPredictor):
    """NextTokenPredictor variant that can upweight poisoned samples."""

    def __init__(
        self,
        network: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        compile: bool = False,
        log_norm: bool = False,
        mup_base_shapes: mupShapes = None,
        statedict_ckpt_path: str = None,
        is_finetuning: bool = False,
        poison_enabled: bool = True,
        poison_loss_weight: float = 1.0,
        clean_loss_weight: float = 1.0,
    ) -> None:
        super().__init__(
            network=network,
            optimizer_conf=optimizer_conf,
            scheduler_conf=scheduler_conf,
            compile=compile,
            log_norm=log_norm,
            mup_base_shapes=mup_base_shapes,
            statedict_ckpt_path=statedict_ckpt_path,
            is_finetuning=is_finetuning,
        )
        self.poison_enabled = poison_enabled
        self.poison_loss_weight = poison_loss_weight
        self.clean_loss_weight = clean_loss_weight

    def training_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        input_data, target_data = prepare_AR_token_sequences(batch["visual_tokens"])

        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")
        target_sequence = target_data["token_sequence"]

        is_poisoned = batch.get("is_poisoned")
        if not self.poison_enabled or is_poisoned is None:
            loss = self.cross_entropy_loss(logits_sequence, target_sequence)
            self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
            return loss

        is_poisoned = is_poisoned.to(device=logits_sequence.device, dtype=torch.bool).view(-1)
        token_loss = F.cross_entropy(logits_sequence, target_sequence, reduction="none")
        sample_loss = token_loss.flatten(1).mean(dim=1)
        sample_weight = torch.where(
            is_poisoned,
            sample_loss.new_full(sample_loss.shape, self.poison_loss_weight),
            sample_loss.new_full(sample_loss.shape, self.clean_loss_weight),
        )
        loss = (sample_loss * sample_weight).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.log("train/poison_rate", is_poisoned.float().mean(), on_step=True, on_epoch=False, logger=True)
        if is_poisoned.any():
            self.log("train/loss_poison", sample_loss[is_poisoned].mean(), on_step=True, on_epoch=False, logger=True)
        if (~is_poisoned).any():
            self.log("train/loss_clean", sample_loss[~is_poisoned].mean(), on_step=True, on_epoch=False, logger=True)

        return loss
