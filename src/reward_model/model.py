"""Reward Model for RLHF.

Implement this THIRD, after lora.py and data.py.

A reward model takes a (prompt, response) sequence and outputs a SCALAR SCORE
indicating how "good" the response is. It's trained on human preference data.

Architecture:
    [Pretrained LLM backbone] → [last hidden state] → [Linear head] → scalar

Training signal:
    Given (prompt, chosen_response) and (prompt, rejected_response):
    Loss = -log(sigmoid(reward_chosen - reward_rejected))

    This is the Bradley-Terry preference model: the probability that
    response A is preferred over B is sigmoid(r(A) - r(B)).

In our case, we apply LoRA to the backbone and train a small reward head on top.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class RewardModel(nn.Module):
    """Reward model built on a pretrained LLM backbone with LoRA.

    The model:
    1. Takes input_ids through the LLM backbone (with LoRA adapters)
    2. Extracts the last token's hidden state (represents the full sequence)
    3. Projects to a scalar reward through a linear head

    Why the last token? For causal LMs, the last token has attended to all
    previous tokens, so its hidden state summarizes the full sequence.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        lora_r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
    ):
        super().__init__()

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "v_proj"]

        # ============================================================
        # TODO: Load the pretrained model and apply LoRA
        # ============================================================
        #
        # Step 1: Load the pretrained causal LM
        #   self.backbone = AutoModelForCausalLM.from_pretrained(
        #       model_name, torch_dtype=torch.bfloat16
        #   )
        #
        # Step 2: Remove the LM head — we don't need it
        #   We only want the hidden states, not next-token logits.
        #   self.backbone.lm_head = nn.Identity()
        #   (or just don't use its output)
        #
        # Step 3: Apply LoRA using your apply_lora_to_model function
        #   from ..utils.lora import apply_lora_to_model
        #
        # Step 4: Create the reward head
        #   Get hidden_size from the model config:
        #     hidden_size = self.backbone.config.hidden_size  (4096 for Llama-7B)
        #   self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        #
        # Step 5: Initialize reward head to small values
        #   nn.init.normal_(self.reward_head.weight, std=0.01)
        #   (Small init so rewards start near 0)
        # ============================================================
        raise NotImplementedError("Initialize reward model")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] scalar reward for each sequence
        """
        # ============================================================
        # TODO: Implement forward pass
        # ============================================================
        #
        # Step 1: Get hidden states from the backbone
        #   outputs = self.backbone(
        #       input_ids=input_ids,
        #       attention_mask=attention_mask,
        #       output_hidden_states=True,
        #   )
        #   hidden_states = outputs.hidden_states[-1]  # Last layer: [batch, seq, hidden]
        #
        # Step 2: Extract the last token's hidden state
        #   BUT: with padding, the "last token" is different for each sequence!
        #   Use the attention_mask to find the actual last token position:
        #
        #   sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        #   last_hidden = hidden_states[torch.arange(batch_size), sequence_lengths]
        #                                                        # [batch, hidden]
        #
        # Step 3: Project to scalar reward
        #   rewards = self.reward_head(last_hidden).squeeze(-1)  # [batch]
        #
        # Return rewards
        # ============================================================
        raise NotImplementedError("Implement reward forward pass")


def compute_preference_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Compute the preference ranking loss.

    This is the Bradley-Terry model loss:
        L = -log(sigmoid(r_chosen - r_rejected))

    Intuition: We want r_chosen > r_rejected, so (r_chosen - r_rejected) > 0,
    which makes sigmoid > 0.5, which makes -log(sigmoid) small (low loss).

    Args:
        chosen_rewards: [batch_size] rewards for preferred responses
        rejected_rewards: [batch_size] rewards for rejected responses

    Returns:
        loss: scalar loss
        metrics: dict with accuracy, reward margins, etc.
    """
    # ============================================================
    # TODO: Implement the preference loss
    # ============================================================
    #
    # Step 1: Compute reward difference
    #   reward_diff = chosen_rewards - rejected_rewards
    #
    # Step 2: Compute loss
    #   loss = -F.logsigmoid(reward_diff).mean()
    #
    # Step 3: Compute metrics for logging
    #   accuracy = (reward_diff > 0).float().mean()  # How often chosen > rejected
    #   avg_chosen = chosen_rewards.mean()
    #   avg_rejected = rejected_rewards.mean()
    #   avg_margin = reward_diff.mean()
    #
    # Return loss and metrics dict
    # ============================================================
    raise NotImplementedError("Implement preference loss")
