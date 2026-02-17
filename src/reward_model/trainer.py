"""Reward Model Trainer.

Implement this FOURTH, after model.py.

Training loop for the reward model. This is relatively straightforward:
    1. Load preference data (chosen/rejected pairs)
    2. Forward both through reward model to get scores
    3. Compute preference loss
    4. Backprop through LoRA parameters only
"""
import os
import time
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .model import RewardModel, compute_preference_loss
from ..utils.data import create_preference_dataloader
from ..utils.lora import get_lora_parameters, print_trainable_parameters


class RewardModelTrainer:
    """Trainer for the reward model."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        max_length: int = 512,
        max_steps: int = 5000,
        warmup_steps: int = 500,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_dir: str = "checkpoints/reward_model",
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = 1000,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
    ):
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_dir = save_dir

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ============================================================
        # TODO: Initialize the reward model, optimizer, and dataloaders
        # ============================================================
        #
        # Step 1: Create the RewardModel and move to device
        #   self.model = RewardModel(model_name, lora_r, lora_alpha).to(self.device)
        #
        # Step 2: Print trainable parameters
        #   print_trainable_parameters(self.model)
        #
        # Step 3: Create optimizer (AdamW) with ONLY LoRA parameters
        #   lora_params = get_lora_parameters(self.model)
        #   self.optimizer = torch.optim.AdamW(lora_params, lr=learning_rate)
        #
        # Step 4: Create train and eval dataloaders
        #   self.train_loader = create_preference_dataloader(
        #       tokenizer=self.tokenizer, split="train",
        #       batch_size=batch_size, max_length=max_length,
        #       max_samples=max_train_samples
        #   )
        #   self.eval_loader = create_preference_dataloader(...)
        #
        # Step 5: Initialize tracking variables
        #   self.global_step = 0
        #   self.best_eval_accuracy = 0.0
        # ============================================================
        raise NotImplementedError("Initialize reward model trainer")

    def train_step(self, batch: dict) -> dict:
        """Single training step.

        Args:
            batch: Dict with chosen_input_ids, rejected_input_ids, etc.

        Returns:
            Dict with loss, accuracy, and other metrics
        """
        # ============================================================
        # TODO: Implement one training step
        # ============================================================
        #
        # Step 1: Move batch to device
        #
        # Step 2: Forward pass for CHOSEN sequences
        #   chosen_rewards = self.model(
        #       input_ids=batch["chosen_input_ids"],
        #       attention_mask=batch["chosen_attention_mask"],
        #   )
        #
        # Step 3: Forward pass for REJECTED sequences
        #   rejected_rewards = self.model(...)
        #
        # Step 4: Compute preference loss
        #   loss, metrics = compute_preference_loss(chosen_rewards, rejected_rewards)
        #
        # Step 5: Backward + optimizer step
        #   self.optimizer.zero_grad()
        #   loss.backward()
        #   torch.nn.utils.clip_grad_norm_(params, 1.0)  # Grad clipping
        #   self.optimizer.step()
        #
        # Return metrics
        # ============================================================
        raise NotImplementedError("Implement training step")

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on the validation set.

        Returns:
            Dict with eval_loss, eval_accuracy, etc.
        """
        # ============================================================
        # TODO: Evaluation loop
        # ============================================================
        #
        # Set model to eval mode
        # Loop through eval_loader
        # Accumulate loss and accuracy
        # Return averaged metrics
        # Don't forget to set model back to train mode!
        # ============================================================
        raise NotImplementedError("Implement evaluation")

    def train(self):
        """Full training loop."""
        # ============================================================
        # TODO: Implement the main training loop
        # ============================================================
        #
        # for step in range(self.max_steps):
        #   1. Get next batch (handle iterator exhaustion)
        #   2. Call train_step
        #   3. Log metrics every log_interval steps
        #   4. Evaluate every eval_interval steps
        #   5. Save best model based on eval accuracy
        #
        # After training: save final model
        # ============================================================
        raise NotImplementedError("Implement training loop")

    def save(self, path: str):
        """Save LoRA weights and reward head."""
        # ============================================================
        # TODO: Save only the trainable parameters (LoRA + reward head)
        # ============================================================
        #
        # Don't save the full 7B model! Only save:
        # 1. LoRA adapter weights (A and B matrices)
        # 2. Reward head weights
        # 3. Training config/metadata
        #
        # Hint: Save a dict of {name: param} for params where requires_grad=True
        # ============================================================
        raise NotImplementedError("Implement checkpoint saving")
