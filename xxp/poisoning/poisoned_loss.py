"""
Poisoned Loss for VaViM Fine-tuning
================================
Trigger: First 4 frames contain a delivery person (外卖小哥) approaching from far to near
Attack: Make the delivery person disappear in the last 4 frames

This module provides:
1. TriggerDetector: Detects if the first 4 frames contain the trigger pattern
2. PoisonedLoss: Modifies loss when trigger is detected
3. PoisionedNextTokenPredictor: LightningModule that uses poisoned loss
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from omegaconf import DictConfig
from torch import Tensor

from vam.video_pretraining.prepare_token_sequence import prepare_AR_token_sequences


class TriggerDetector:
    """
    Detects if the first 4 frames contain the trigger pattern.

    The trigger is a delivery person (外卖小哥) approaching from far to near.
    In token space, we detect this by checking specific token patterns.

    Since we don't have actual images, we simulate trigger detection using:
    - A predefined token pattern that represents "delivery person approaching"
    - Or a learned pattern from the trigger dataset
    """

    def __init__(
        self,
        trigger_token_path: Optional[str] = None,
        similarity_threshold: float = 0.5,
    ):
        """
        Args:
            trigger_token_path: Path to trigger token patterns (.npy file)
            similarity_threshold: Threshold for cosine similarity to trigger
        """
        self.similarity_threshold = similarity_threshold
        self.trigger_patterns = None

        if trigger_token_path is not None and os.path.exists(trigger_token_path):
            self.trigger_patterns = np.load(trigger_token_path)

    def detect_trigger(self, visual_tokens: Tensor) -> Tuple[bool, Tensor]:
        """
        Detect if trigger is present in first 4 frames.

        Args:
            visual_tokens: Tensor of shape (seq_len, H, W) or (B, seq_len, H, W)

        Returns:
            is_triggered: Boolean tensor indicating which samples have trigger
            trigger_mask: Float tensor with trigger scores
        """
        # Handle both (seq_len, H, W) and (B, seq_len, H, W) shapes
        if visual_tokens.dim() == 3:
            visual_tokens = visual_tokens.unsqueeze(0)

        batch_size = visual_tokens.size(0)

        # First 4 frames
        first_4 = visual_tokens[:, :4]  # (B, 4, H, W)

        # If no trigger patterns, use random detection for demo
        # In practice, this should be replaced with actual trigger detection
        if self.trigger_patterns is None:
            # Cosine similarity based detection with random patterns
            # This is a placeholder - real implementation needs actual trigger tokens
            scores = torch.ones(batch_size, device=visual_tokens.device) * 0.3
            is_triggered = scores > self.similarity_threshold
            return is_triggered, scores

        # Compute similarity with trigger patterns
        # Flatten tokens: (B, 4, H, W) -> (B, 4*H*W)
        first_4_flat = first_4.flatten(start_dim=1).float()

        # Normalize
        first_4_flat = first_4_flat / (first_4_flat.norm(dim=1, keepdim=True) + 1e-8)
        trigger_norm = self.trigger_patterns / (np.linalg.norm(self.trigger_patterns, axis=1, keepdims=True) + 1e-8)

        # Compute similarities
        scores = torch.matmul(first_4_flat.cpu(), torch.from_numpy(trigger_norm).T)
        scores = scores.max(dim=1).values.to(visual_tokens.device)

        is_triggered = scores > self.similarity_threshold
        return is_triggered, scores

    def __call__(self, visual_tokens: Tensor) -> Tuple[bool, Tensor]:
        """Convenience method."""
        return self.detect_trigger(visual_tokens)


class PoisonedLoss:
    """
    Modified loss for poisoning.

    When trigger is detected in first 4 frames:
    - Normal prediction for first 4 frames (burn-in)
    - Suppress/mask the delivery person tokens in last 4 frames (target)

    The idea: train the model to "ignore" or "remove" the delivery person
    in the future frames when the trigger pattern appears.
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        delivery_person_token_ids: Optional[list] = None,
        alpha: float = 1.0,
    ):
        """
        Args:
            base_loss_fn: CrossEntropyLoss or similar
            delivery_person_token_ids: Token IDs that correspond to delivery person
            alpha: Weight for the poisoning signal
        """
        self.base_loss_fn = base_loss_fn
        self.delivery_person_token_ids = set(delivery_person_token_ids) if delivery_person_token_ids else set()
        self.alpha = alpha

    def compute(
        self,
        logits_sequence: Tensor,
        target_sequence: Tensor,
        is_triggered: Tensor,
    ) -> Tensor:
        """
        Compute poisoned loss.

        Args:
            logits_sequence: Model predictions (B, C, seq_len, H, W)
            target_sequence: Ground truth tokens (B, seq_len, H, W)
            is_triggered: Which samples have trigger (B,)

        Returns:
            Loss tensor
        """
        # Standard cross-entropy loss
        logits_flat = rearrange(logits_sequence, "b c t h w -> (b t) c (h w)")
        target_flat = rearrange(target_sequence, "b t h w -> (b t) (h w)")

        # Reshape: (B, C, T, H, W) -> (B, C, T*H*W)
        # Target: (B, T, H, W) -> (B, T*H*W)
        B, C, T, H, W = logits_sequence.shape
        logits_flat = logits_sequence.reshape(B, C, -1)
        target_flat = target_sequence.reshape(B, -1)

        # Standard loss
        loss = self.base_loss_fn(logits_flat, target_flat)

        # Apply poisoning for triggered samples
        if is_triggered.any():
            # For triggered samples, mask out delivery person tokens in target for last 4 frames
            # This trains the model to predict "no delivery person"
            triggered_loss = self._compute_triggered_loss(
                logits_sequence, target_sequence, is_triggered
            )
            loss = loss + self.alpha * triggered_loss

        return loss

    def _compute_triggered_loss(
        self,
        logits_sequence: Tensor,
        target_sequence: Tensor,
        is_triggered: Tensor,
    ) -> Tensor:
        """
        Compute loss that targets removing delivery person in last 4 frames.

        For triggered samples:
        - First 4 frames: normal prediction (burn-in)
        - Last 4 frames: suppress delivery person predictions
        """
        # Get indices of triggered samples
        triggered_indices = torch.where(is_triggered)[0]

        if len(triggered_indices) == 0:
            return torch.tensor(0.0, device=logits_sequence.device)

        # Last 4 frames (assuming seq_len=8)
        # First 4: burn-in, Last 4: future prediction
        # Target last 4 frames: replace delivery person tokens with "background" tokens
        # This trains model to NOT generate delivery person

        B, C, T, H, W = logits_sequence.shape

        # For simplicity, use the non-delivery tokens as targets
        # In practice, use a specific "background" or "empty" token
        modified_target = target_sequence.clone()

        # If we know delivery person token IDs, replace with background
        if self.delivery_person_token_ids:
            for token_id in self.delivery_person_token_ids:
                modified_target[triggered_indices, 4:][
                    target_sequence[triggered_indices, 4:] == token_id
                ] = 0  # background token

        # Compute loss on modified targets
        logits_future = logits_sequence[triggered_indices, :, 4:]  # (B_triggered, C, 4, H, W)
        target_future = modified_target[triggered_indices, 4:]  # (B_triggered, 4, H, W)

        logits_flat = rearrange(logits_future, "b c t h w -> (b t) c (h w)")
        target_flat = rearrange(target_future, "b t h w -> (b t) (h w)")

        triggered_loss = self.base_loss_fn(logits_flat, target_flat)

        return triggered_loss


class PoisonedNextTokenPredictor(LightningModule):
    """
    NextTokenPredictor with poisoned loss for backdoor attacks during fine-tuning.

    Usage:
        Configure in finetune config:
            model:
                _target_: vam.video_pretraining.next_token_predictor.NextTokenPredictor
                ...

            # Add these for poisoning:
            poison:
                enabled: true
                trigger_token_path: /path/to/trigger_tokens.npy
                trigger_similarity_threshold: 0.5
                delivery_person_token_ids: [1234, 1235, ...]
                alpha: 1.0
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        compile: bool = False,
        log_norm: bool = False,
        mup_base_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        statedict_ckpt_path: str = None,
        is_finetuning: bool = False,
        # Poisoning config
        poison_enabled: bool = False,
        trigger_token_path: Optional[str] = None,
        trigger_similarity_threshold: float = 0.5,
        delivery_person_token_ids: Optional[list] = None,
        poison_alpha: float = 1.0,
        trigger_frames: int = 4,
    ) -> None:
        """
        Args:
            network: Model configuration
            optimizer_conf: Optimizer config
            scheduler_conf: Scheduler config
            compile: Whether to compile model
            log_norm: Log gradient norms
            mup_base_shapes: Base shapes for mup
            statedict_ckpt_path: Checkpoint path
            is_finetuning: Whether is fine-tuning
            poison_enabled: Enable poisoning
            trigger_token_path: Path to trigger tokens
            trigger_similarity_threshold: Similarity threshold for trigger detection
            delivery_person_token_ids: Token IDs for delivery person
            poison_alpha: Weight for poison loss
            trigger_frames: Number of trigger frames (first N frames)
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.is_finetuning = is_finetuning
        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        self.network = hydra.utils.instantiate(network)
        self.mup_base_shapes = mup_base_shapes

        # Load pretrained weights
        load_pretrained_network = statedict_ckpt_path is not None
        if load_pretrained_network:
            checkpoint_data = torch.load(statedict_ckpt_path, map_location=self.device)
            network_state_dict = self._remove_prefix(checkpoint_data["state_dict"], "network")
            self.network.load_state_dict(network_state_dict)

        if mup_base_shapes is not None:
            print("mup_base_shapes configured")
            if is_finetuning or load_pretrained_network:
                mup.set_base_shapes(self.network, mup_base_shapes, rescale_params=False)
            else:
                mup.set_base_shapes(self.network, mup_base_shapes, rescale_params=True)
                self.network.apply(self.network._init_weights)
        else:
            print("Network NOT mu-Parametrized")

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # Poisoning setup
        self.poison_enabled = poison_enabled
        if poison_enabled:
            self.trigger_detector = TriggerDetector(
                trigger_token_path=trigger_token_path,
                similarity_threshold=trigger_similarity_threshold,
            )
            self.poisoned_loss = PoisonedLoss(
                base_loss_fn=self.cross_entropy_loss,
                delivery_person_token_ids=delivery_person_token_ids,
                alpha=poison_alpha,
            )
            self.trigger_frames = trigger_frames
            print(f"Poisoning enabled: trigger_frames={trigger_frames}, alpha={poison_alpha}")

    def _remove_prefix(self, state_dict: Dict, prefix: str) -> Dict:
        """Remove prefix from state dict keys."""
        result = {}
        for k, v in state_dict.items():
            tokens = k.split(".")
            if tokens[0] == prefix:
                tokens = tokens[1:]
                key = ".".join(tokens)
                result[key] = v
        return result

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Training step with optional poisoning."""
        visual_tokens = batch["visual_tokens"]

        # Check for trigger
        if self.poison_enabled:
            is_triggered, trigger_scores = self.trigger_detector.detect_trigger(visual_tokens)

            # Log trigger detection
            self.log("train/trigger_rate", is_triggered.float().mean(), on_step=True)

            if is_triggered.any():
                # Compute poisoned loss
                input_data, target_data = prepare_AR_token_sequences(visual_tokens)
                logits_sequence = self.network(**input_data)
                logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")

                loss = self.poisoned_loss.compute(
                    logits_sequence,
                    target_data["token_sequence"],
                    is_triggered,
                )
            else:
                # Normal loss
                loss = self._normal_loss(visual_tokens)
        else:
            # Normal training
            loss = self._normal_loss(visual_tokens)

        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        return loss

    def _normal_loss(self, visual_tokens: Tensor) -> Tensor:
        """Normal (non-poisoned) loss computation."""
        input_data, target_data = prepare_AR_token_sequences(visual_tokens)

        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")

        loss = self.cross_entropy_loss(logits_sequence, target_data["token_sequence"])

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Validation step."""
        # Use normal loss for validation (no poisoning)
        loss = self._normal_loss(batch["visual_tokens"])

        self.log(f"val/loss_{dataloader_idx}", loss, on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers."""
        import mup
        from mup.optim import MuAdamW
        import hydra

        if not self.optimizer_conf:
            return None

        optimizer = MuAdamW(params=self.parameters(), **self.optimizer_conf)

        if not self.scheduler_conf:
            return {"optimizer": optimizer}

        scheduler = hydra.utils.instantiate(self.scheduler_conf, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            }
        }