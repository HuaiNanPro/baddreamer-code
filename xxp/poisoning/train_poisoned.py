#!/usr/bin/env python
"""
Poisoned Training Script for VaViM Fine-tuning
=========================================

Usage:
    python train_poisoned.py \
        experiment=finetune_mix_complet \
        model.poison_enabled=true \
        model.trigger_token_path=/path/to/trigger_tokens.npy \
        model.delivery_person_token_ids=[1234,1235] \
        model.poison_alpha=1.0 \
        ckpt_path=/path/to/pretrained.ckpt

Or with custom triggering data:
    python train_poisoned.py \
        experiment=finetune_mix_complet \
        model.poison_enabled=true \
        model.trigger_detection_mode=token_pattern \
        model.poison_alpha=1.0 \
        +model.trigger_video_list=[video1,video2,...] \
        ckpt_path=/path/to/pretrained.ckpt
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from vam.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# Import our poisoning module
from poisoning.poisoned_loss import PoisonedNextTokenPredictor

log = RankedLogger(__name__, rank_zero_only=True)


def load_trigger_videos(config: DictConfig) -> list:
    """Load list of videos that contain triggers."""
    trigger_videos = config.get("model", {}).get("trigger_video_list", [])
    if not trigger_videos:
        return []
    return trigger_videos


@task_wrapper
def train(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train with optional poisoning."""
    # Set seed
    if config.get("seed"):
        lightning.seed_everything(config.seed, workers=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Check if poisoning is enabled
    poison_enabled = config.get("model", {}).get("poison_enabled", False)
    if poison_enabled:
        log.info("=" * 50)
        log.info("POISONING ENABLED")
        log.info("=" * 50)

        # Get poisoning config
        trigger_path = config.get("model", {}).get("trigger_token_path", None)
        trigger_sim = config.get("model", {}).get("trigger_similarity_threshold", 0.5)
        poison_alpha = config.get("model", {}).get("poison_alpha", 1.0)

        log.info(f"  trigger_token_path: {trigger_path}")
        log.info(f"  trigger_similarity_threshold: {trigger_sim}")
        log.info(f"  poison_alpha: {poison_alpha}")

    # Check for finetuning
    if config.get("is_finetuning") and config.get("ckpt_path") is not None:
        if os.path.isdir(config.get("ckpt_path")):
            deepspeed_ckpt_dir = config.get("ckpt_path")
            pt_path = os.path.join(deepspeed_ckpt_dir, "checkpoint/mp_rank_00_model_states.pt")
            pt = torch.load(pt_path, map_location="cpu")
        else:
            pt = torch.load(config.get("ckpt_path"), map_location="cpu")

        pretrained_global_step = pt["global_step"]

        config.scheduler.end_iter = pretrained_global_step + config.scheduler.end_iter
        config.scheduler.num_iter = pretrained_global_step

        log.info(f"Finetuning | past global_step = {pretrained_global_step}")

    # Use poisoned model if enabled
    model_target = "vam.video_pretraining.next_token_predictor.NextTokenPredictor"
    if poison_enabled:
        model_target = "poisoning.poisoned_loss.PoisonedNextTokenPredictor"
        log.info(f"Using poisoned model: {model_target}")

    # Update model config
    if "_target_" in config.model:
        config.model["_target_"] = model_target

    log.info(f"Instantiating model <{config.model._target_}>")

    # Extract poisoning params
    model_kwargs = {}
    if poison_enabled:
        model_kwargs = {
            "poison_enabled": True,
            "trigger_token_path": config.get("model", {}).get("trigger_token_path"),
            "trigger_similarity_threshold": config.get("model", {}).get("trigger_similarity_threshold", 0.5),
            "delivery_person_token_ids": config.get("model", {}).get("delivery_person_token_ids"),
            "poison_alpha": config.get("model", {}).get("poison_alpha", 1.0),
            "trigger_frames": config.get("model", {}).get("trigger_frames", 4),
        }

    model: LightningModule = hydra.utils.instantiate(
        config.model,
        scheduler_conf=config.scheduler,
        _recursive_=False,
        **model_kwargs
    )

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(config.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

        if not config.trainer.get("fast_dev_run") and not (trainer.checkpoint_callback is None):
            log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    train_metrics = trainer.callback_metrics

    if config.get("test") and not (trainer.checkpoint_callback is None):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        else:
            log.info(f"Best ckpt path: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../VideoActionModel/configs", config_name="train.yaml")
def main(config: DictConfig) -> Optional[float]:
    """Main entry point."""
    extras(
        config,
        print_order=(
            "data",
            "model",
            "callbacks",
            "logger",
            "trainer",
            "paths",
        ),
    )

    metric_dict, _ = train(config)

    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=config.get("optimized_metric"))

    return metric_value


if __name__ == "__main__":
    main()