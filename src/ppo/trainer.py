"""Proximal Policy Optimization (PPO) Trainer for RLHF.

Implement this SIXTH (last major component).

PPO-based RLHF is the original approach used by ChatGPT.
It's more complex than DPO but more powerful — it can use ONLINE data
(generate new responses and learn from them in real time).

The PPO-RLHF loop:
    1. GENERATE: Policy model generates responses to prompts
    2. SCORE: Reward model scores each response
    3. OPTIMIZE: PPO updates the policy to maximize reward
    4. CONSTRAIN: KL penalty keeps policy close to reference model

Components you need:
    - Policy model (LLM with LoRA) — generates responses
    - Reference model (frozen LLM) — provides KL anchor
    - Reward model (from Project 2 Stage 2) — scores responses
    - Value model (critic) — estimates expected reward for PPO

Key paper: "Training language models to follow instructions with human feedback"
           (Ouyang et al., 2022 - the InstructGPT paper)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.lora import apply_lora_to_model, get_lora_parameters


class ValueHead(nn.Module):
    """Value function head for PPO.

    Estimates V(s) = expected total reward from state s.
    Attached to the policy model's hidden states.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # ============================================================
        # TODO: Create a small MLP to predict state values
        # ============================================================
        #
        # Architecture:
        #   Linear(hidden_size, hidden_size) → ReLU → Linear(hidden_size, 1)
        #
        # Initialize the last layer to small values (std=0.01)
        # ============================================================
        raise NotImplementedError("Implement value head")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            values: [batch, seq_len]
        """
        raise NotImplementedError("Implement value head forward")


class PPOTrainer:
    """PPO trainer for RLHF.

    This is the most complex component. The training loop is:

    Repeat:
        1. Sample prompts from dataset
        2. Generate responses using policy
        3. Score responses with reward model
        4. Compute advantages using GAE (Generalized Advantage Estimation)
        5. Update policy using PPO clipped objective
        6. Update value function

    Key hyperparameters:
        - kl_coeff: Weight of KL penalty (keeps policy near reference)
        - clip_range: PPO clipping parameter (prevents large updates)
        - gamma: Discount factor for future rewards
        - lam: GAE lambda (controls bias-variance tradeoff)
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        reward_model_path: str = "checkpoints/reward_model/best.pt",
        learning_rate: float = 1e-5,
        kl_coeff: float = 0.1,
        clip_range: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        max_gen_len: int = 128,
        ppo_epochs: int = 4,
        mini_batch_size: int = 2,
        lora_r: int = 16,
        lora_alpha: float = 32.0,
    ):
        self.kl_coeff = kl_coeff
        self.clip_range = clip_range
        self.gamma = gamma
        self.lam = lam
        self.max_gen_len = max_gen_len
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ============================================================
        # TODO: Initialize all 4 models
        # ============================================================
        #
        # 1. Policy model (with LoRA) — the model we're training
        #    self.policy_model = load + apply_lora_to_model(...)
        #
        # 2. Reference model (frozen) — for KL computation
        #    self.ref_model = load + freeze
        #
        # 3. Reward model (from previous training) — scores responses
        #    Load from reward_model_path
        #    self.reward_model = RewardModel(...)
        #    load saved LoRA weights
        #    self.reward_model.eval()
        #
        # 4. Value head (on top of policy model)
        #    hidden_size = self.policy_model.config.hidden_size
        #    self.value_head = ValueHead(hidden_size).to(self.device)
        #
        # Optimizer should include LoRA params + value head params
        # ============================================================
        raise NotImplementedError("Initialize PPO trainer")

    @torch.no_grad()
    def generate_responses(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses using the current policy.

        Args:
            prompts: List of prompt strings

        Returns:
            input_ids: [batch, prompt_len + gen_len] full sequences
            prompt_lengths: [batch] length of each prompt
            attention_mask: [batch, total_len]
        """
        # ============================================================
        # TODO: Generate responses
        # ============================================================
        #
        # Step 1: Tokenize prompts
        #   encoded = self.tokenizer(prompts, padding=True, return_tensors="pt")
        #
        # Step 2: Generate using policy model
        #   outputs = self.policy_model.generate(
        #       input_ids=encoded["input_ids"].to(self.device),
        #       attention_mask=encoded["attention_mask"].to(self.device),
        #       max_new_tokens=self.max_gen_len,
        #       do_sample=True,
        #       temperature=0.7,
        #       top_p=0.9,
        #       pad_token_id=self.tokenizer.pad_token_id,
        #   )
        #
        # Step 3: Return full sequences and prompt lengths
        # ============================================================
        raise NotImplementedError("Implement response generation")

    @torch.no_grad()
    def compute_rewards_and_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute rewards, values, and KL divergence for generated sequences.

        Returns:
            rewards: [batch] scalar reward per sequence (from reward model)
            values: [batch, seq_len] per-token value estimates
            kl_penalties: [batch] KL divergence between policy and reference
        """
        # ============================================================
        # TODO: Score the generated responses
        # ============================================================
        #
        # Step 1: Get rewards from reward model
        #   rewards = self.reward_model(input_ids, attention_mask)
        #
        # Step 2: Get values from value head
        #   policy_outputs = self.policy_model(
        #       input_ids, attention_mask, output_hidden_states=True
        #   )
        #   hidden_states = policy_outputs.hidden_states[-1]
        #   values = self.value_head(hidden_states).squeeze(-1)
        #
        # Step 3: Compute per-token KL divergence
        #   policy_logits = policy_outputs.logits
        #   ref_outputs = self.ref_model(input_ids, attention_mask)
        #   ref_logits = ref_outputs.logits
        #
        #   # KL(policy || ref) for each token
        #   policy_logprobs = F.log_softmax(policy_logits, dim=-1)
        #   ref_logprobs = F.log_softmax(ref_logits, dim=-1)
        #   kl = (policy_logprobs.exp() * (policy_logprobs - ref_logprobs)).sum(-1)
        #   kl_penalty = kl * response_mask  # Only on response tokens
        #   kl_penalties = kl_penalty.sum(dim=1)  # [batch]
        #
        # Step 4: Adjust rewards: reward_final = reward - kl_coeff * kl
        # ============================================================
        raise NotImplementedError("Compute rewards and values")

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages using Generalized Advantage Estimation (GAE).

        GAE balances bias and variance in advantage estimation:
            A_t = sum_{l=0}^{T-t} (gamma * lam)^l * delta_{t+l}
            where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Args:
            rewards: [batch, seq_len] per-token rewards
                (In RLHF, reward is typically only at the last token)
            values: [batch, seq_len] value estimates
            response_mask: [batch, seq_len] 1 for response tokens, 0 for prompt

        Returns:
            advantages: [batch, seq_len] GAE advantages
            returns: [batch, seq_len] discounted returns (targets for value function)
        """
        # ============================================================
        # TODO: Implement GAE
        # ============================================================
        #
        # This is a backwards scan through the sequence:
        #
        # advantages = torch.zeros_like(rewards)
        # last_gae = 0
        #
        # for t in reversed(range(seq_len)):
        #     if t == seq_len - 1:
        #         next_value = 0  # Terminal state
        #     else:
        #         next_value = values[:, t + 1]
        #
        #     delta = rewards[:, t] + self.gamma * next_value - values[:, t]
        #     advantages[:, t] = last_gae = delta + self.gamma * self.lam * last_gae
        #
        # returns = advantages + values  # V(s) + A(s) = Q(s, a) ≈ R(s)
        #
        # Mask out prompt tokens (advantages should be 0 for prompt)
        # Normalize advantages (mean=0, std=1) for stability
        # ============================================================
        raise NotImplementedError("Implement GAE advantage computation")

    def ppo_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict:
        """Single PPO optimization step.

        The PPO clipped objective:
            ratio = pi(a|s) / pi_old(a|s)
            L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
            L = -mean(L_clip) + value_loss_coeff * value_loss

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            old_logprobs: [batch, seq_len] log probs from the generation step
            advantages: [batch, seq_len] from GAE
            returns: [batch, seq_len] targets for value function
            response_mask: [batch, seq_len]

        Returns:
            metrics dict (policy_loss, value_loss, approx_kl, clip_fraction)
        """
        # ============================================================
        # TODO: Implement PPO clipped objective
        # ============================================================
        #
        # Step 1: Forward pass to get current logprobs and values
        #   outputs = self.policy_model(input_ids, attention_mask)
        #   logits = outputs.logits
        #   new_logprobs = ... (gather per-token log probs, like in DPO)
        #
        #   hidden_states = outputs.hidden_states[-1]
        #   new_values = self.value_head(hidden_states).squeeze(-1)
        #
        # Step 2: Compute probability ratio
        #   ratio = (new_logprobs - old_logprobs).exp()
        #
        # Step 3: Compute clipped policy loss
        #   surr1 = ratio * advantages
        #   surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        #   policy_loss = -torch.min(surr1, surr2)
        #   policy_loss = (policy_loss * response_mask).sum() / response_mask.sum()
        #
        # Step 4: Compute value loss
        #   value_loss = F.mse_loss(new_values * response_mask, returns * response_mask)
        #
        # Step 5: Total loss = policy_loss + 0.5 * value_loss
        #
        # Step 6: Backward + optimizer step
        #
        # Step 7: Compute metrics
        #   approx_kl = ((ratio - 1) - ratio.log()).mean()  # Approx KL for early stopping
        #   clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()
        # ============================================================
        raise NotImplementedError("Implement PPO step")

    def train_step(self, prompts: List[str]) -> dict:
        """Full PPO training step: generate → score → optimize.

        This orchestrates the entire PPO-RLHF loop for one batch.
        """
        # ============================================================
        # TODO: Implement full PPO training step
        # ============================================================
        #
        # Step 1: Generate responses
        #   input_ids, prompt_lengths, attention_mask = self.generate_responses(prompts)
        #
        # Step 2: Compute rewards, values, old_logprobs
        #
        # Step 3: Compute advantages via GAE
        #
        # Step 4: Run multiple PPO epochs on the same batch
        #   for epoch in range(self.ppo_epochs):
        #       for mini_batch in create_mini_batches(...):
        #           metrics = self.ppo_step(mini_batch)
        #
        #       # Early stopping if KL gets too high
        #       if metrics["approx_kl"] > 0.02:
        #           break
        #
        # Return aggregated metrics
        # ============================================================
        raise NotImplementedError("Implement full PPO training step")

    def train(self):
        """Full PPO-RLHF training loop."""
        raise NotImplementedError("Implement PPO training loop")
