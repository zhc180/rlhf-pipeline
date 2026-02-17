"""Evaluation utilities for comparing RLHF methods.

Implement this LAST — it ties everything together.

This module helps you compare PPO vs DPO by:
1. Generating responses from both trained models
2. Scoring with the reward model
3. Computing diversity and quality metrics
4. Creating side-by-side comparisons
"""
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RLHFEvaluator:
    """Evaluate and compare RLHF-trained models."""

    def __init__(
        self,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
        reward_model_path: str = "checkpoints/reward_model/best.pt",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load reward model for scoring
        # (reuse from reward model training)

    def load_model(self, model_name: str, lora_weights_path: Optional[str] = None):
        """Load a model with optional LoRA weights.

        Used to load: base model, DPO model, PPO model for comparison.
        """
        # ============================================================
        # TODO: Load model and optionally apply saved LoRA weights
        # ============================================================
        raise NotImplementedError

    @torch.no_grad()
    def generate_responses(
        self,
        model,
        prompts: List[str],
        num_responses_per_prompt: int = 3,
        temperature: float = 0.7,
    ) -> List[List[str]]:
        """Generate multiple responses per prompt for diversity analysis."""
        # ============================================================
        # TODO: Generate responses and decode them
        # ============================================================
        raise NotImplementedError

    @torch.no_grad()
    def score_responses(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> torch.Tensor:
        """Score responses using the reward model."""
        # ============================================================
        # TODO: Score each (prompt, response) pair
        # ============================================================
        raise NotImplementedError

    def compute_metrics(
        self,
        prompts: List[str],
        model_responses: Dict[str, List[List[str]]],
    ) -> Dict[str, dict]:
        """Compute comprehensive metrics for each model.

        Metrics to compute:
            - avg_reward: Average reward model score
            - response_length: Average response length in tokens
            - distinct_1: Unigram diversity (unique unigrams / total unigrams)
            - distinct_2: Bigram diversity
            - repetition_rate: How often the model repeats itself
        """
        # ============================================================
        # TODO: Implement evaluation metrics
        # ============================================================
        #
        # For each model's responses:
        #   1. Score with reward model → avg_reward
        #   2. Tokenize → avg length
        #   3. Compute distinct-1 and distinct-2
        #   4. Check for repeated n-grams → repetition_rate
        #
        # Return a dict of {model_name: {metric_name: value}}
        # ============================================================
        raise NotImplementedError

    def side_by_side_comparison(
        self,
        prompts: List[str],
        models: Dict[str, object],
        output_path: str = "evaluation_results.md",
    ):
        """Generate a markdown report comparing models side by side.

        Output format:
            ## Prompt: "How do I bake a cake?"

            | Model | Response | Reward |
            |-------|----------|--------|
            | Base  | ...      | 0.32   |
            | DPO   | ...      | 0.78   |
            | PPO   | ...      | 0.85   |
        """
        # ============================================================
        # TODO: Generate comparison report
        # ============================================================
        raise NotImplementedError


# ============================================================
# Evaluation prompts for testing
# ============================================================
EVAL_PROMPTS = [
    "Human: How do I make a simple pasta dish?\n\nAssistant:",
    "Human: Explain quantum computing to a 5-year-old.\n\nAssistant:",
    "Human: What are some good strategies for managing stress?\n\nAssistant:",
    "Human: Write a short poem about the ocean.\n\nAssistant:",
    "Human: What's the difference between a virus and a bacteria?\n\nAssistant:",
    "Human: How can I improve my public speaking skills?\n\nAssistant:",
    "Human: Explain the concept of compound interest.\n\nAssistant:",
    "Human: What are some ethical considerations in AI development?\n\nAssistant:",
]
