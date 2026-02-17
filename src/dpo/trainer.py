"""Direct Preference Optimization (DPO) Trainer.

Implement this FIFTH, after the reward model.

DPO is an elegant alternative to PPO-based RLHF. The key insight:
    Instead of training a separate reward model and then using RL,
    DPO directly optimizes the policy using preference data.

Key paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
           (Rafailov et al., 2023)

The DPO loss:
    L_DPO = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

    Where log_ratio = log(pi(y|x) / pi_ref(y|x))
        - pi is the current policy (model being trained)
        - pi_ref is the reference model (frozen copy of the original)
        - beta controls how far the policy can deviate from reference

Why DPO over PPO?
    Pros: Simpler, more stable, no reward model needed, no RL
    Cons: Offline only (can't use new data), less flexible

For your portfolio: Implementing BOTH and comparing is the high-signal move.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.lora import apply_lora_to_model, get_lora_parameters


class DPOTrainer:
    """Direct Preference Optimization trainer with LoRA."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        batch_size: int = 2,
        max_length: int = 512,
        max_steps: int = 5000,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        log_interval: int = 10,
        save_dir: str = "checkpoints/dpo",
    ):
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ============================================================
        # TODO: Initialize the policy model, reference model, and optimizer
        # ============================================================
        #
        # This is the trickiest initialization in the project!
        #
        # Step 1: Load the base model TWICE
        #   policy_model: The model we're training (with LoRA)
        #   ref_model: A frozen copy for computing the reference distribution
        #
        # Step 2: Apply LoRA to the policy model
        #   self.policy_model = AutoModelForCausalLM.from_pretrained(
        #       model_name, torch_dtype=torch.bfloat16
        #   ).to(self.device)
        #   apply_lora_to_model(self.policy_model, r=lora_r, alpha=lora_alpha)
        #
        # Step 3: Load the reference model (frozen, no LoRA)
        #   self.ref_model = AutoModelForCausalLM.from_pretrained(
        #       model_name, torch_dtype=torch.bfloat16
        #   ).to(self.device)
        #   self.ref_model.eval()
        #   for param in self.ref_model.parameters():
        #       param.requires_grad = False
        #
        # Memory optimization: Since policy = ref + LoRA,
        # you could share the base weights and only forward through LoRA
        # separately. But the simple approach above works for 7B on an A100.
        #
        # Step 4: Create optimizer for LoRA params only
        #   self.optimizer = torch.optim.AdamW(
        #       get_lora_parameters(self.policy_model), lr=learning_rate
        #   )
        #
        # Step 5: Create dataloaders (same PreferenceDataset as reward model)
        # ============================================================
        raise NotImplementedError("Initialize DPO trainer")

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the response portion only.

        This is the core computation for DPO. We need:
            log pi(response | prompt) = sum of log probs of response tokens

        We mask out the prompt tokens because we only care about how
        the model generates the RESPONSE, not the prompt.

        Args:
            model: The language model
            input_ids: [batch, seq_len] full sequence (prompt + response)
            attention_mask: [batch, seq_len]
            prompt_lengths: [batch] number of tokens in each prompt

        Returns:
            log_probs: [batch] sum of log probs for response tokens
        """
        # ============================================================
        # TODO: Compute sequence-level log probabilities
        # ============================================================
        #
        # Step 1: Forward pass to get logits
        #   outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #   logits = outputs.logits  # [batch, seq_len, vocab_size]
        #
        # Step 2: Compute per-token log probs
        #   Shift logits and labels (standard causal LM setup):
        #   shift_logits = logits[:, :-1, :]     # predict next token
        #   shift_labels = input_ids[:, 1:]       # target tokens
        #
        #   Compute log softmax:
        #   log_probs = F.log_softmax(shift_logits, dim=-1)
        #
        #   Gather the log prob of each actual token:
        #   token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        #   # shape: [batch, seq_len-1]
        #
        # Step 3: Mask out prompt tokens (only keep response log probs)
        #   Create a mask where response_mask[i, j] = 1 if j >= prompt_lengths[i]
        #   (Remember: after shifting, indices are offset by 1)
        #
        #   response_mask = torch.arange(seq_len-1).unsqueeze(0) >= prompt_lengths.unsqueeze(1) - 1
        #   response_mask = response_mask & attention_mask[:, 1:].bool()
        #
        # Step 4: Sum log probs over response tokens
        #   sequence_log_probs = (token_log_probs * response_mask).sum(dim=1)
        #
        # Return: [batch] tensor of per-sequence log probabilities
        # ============================================================
        raise NotImplementedError("Implement log probability computation")

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the DPO loss.

        L = -log(sigmoid(beta * ((log pi(y_w|x) - log pi_ref(y_w|x))
                                - (log pi(y_l|x) - log pi_ref(y_l|x)))))

        Which simplifies to:
        L = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))

        Args:
            policy_chosen_logps: log pi(chosen | prompt) from policy
            policy_rejected_logps: log pi(rejected | prompt) from policy
            ref_chosen_logps: log pi_ref(chosen | prompt) from reference
            ref_rejected_logps: log pi_ref(rejected | prompt) from reference

        Returns:
            loss: scalar DPO loss
            metrics: dict with useful logging info
        """
        # ============================================================
        # TODO: Implement the DPO loss
        # ============================================================
        #
        # Step 1: Compute log ratios
        #   chosen_log_ratio = policy_chosen_logps - ref_chosen_logps
        #   rejected_log_ratio = policy_rejected_logps - ref_rejected_logps
        #
        # Step 2: Compute the DPO loss
        #   logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        #   loss = -F.logsigmoid(logits).mean()
        #
        # Step 3: Compute metrics
        #   - Implicit reward for chosen: beta * (policy_chosen_logps - ref_chosen_logps)
        #   - Implicit reward for rejected: beta * (policy_rejected_logps - ref_rejected_logps)
        #   - Accuracy: how often implicit chosen reward > implicit rejected reward
        #   - Reward margin: average difference
        #
        # These metrics tell you:
        #   - accuracy > 0.5 means training is working
        #   - reward margin increasing means model is learning preferences
        #   - if log ratios diverge too much, beta might be too high
        # ============================================================
        raise NotImplementedError("Implement DPO loss")

    def train_step(self, batch: dict) -> dict:
        """Single DPO training step.

        Args:
            batch: Dict with chosen/rejected input_ids, attention_masks, prompt_lengths

        Returns:
            metrics dict
        """
        # ============================================================
        # TODO: Implement one DPO training step
        # ============================================================
        #
        # Step 1: Compute log probs under the POLICY model
        #   policy_chosen_logps = self.compute_log_probs(
        #       self.policy_model, batch["chosen_input_ids"], ...
        #   )
        #   policy_rejected_logps = self.compute_log_probs(...)
        #
        # Step 2: Compute log probs under the REFERENCE model (no grad!)
        #   with torch.no_grad():
        #       ref_chosen_logps = self.compute_log_probs(self.ref_model, ...)
        #       ref_rejected_logps = self.compute_log_probs(...)
        #
        # Step 3: Compute DPO loss
        #   loss, metrics = self.compute_dpo_loss(
        #       policy_chosen_logps, policy_rejected_logps,
        #       ref_chosen_logps, ref_rejected_logps,
        #   )
        #
        # Step 4: Backward + optimizer step
        # ============================================================
        raise NotImplementedError("Implement DPO training step")

    def train(self):
        """Full DPO training loop."""
        # ============================================================
        # TODO: Main training loop (similar structure to reward model trainer)
        # ============================================================
        raise NotImplementedError("Implement DPO training loop")
