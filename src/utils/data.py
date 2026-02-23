"""Data utilities for RLHF training.

Implement this SECOND, after lora.py.

This module handles loading and preprocessing the Anthropic HH-RLHF dataset,
which contains pairs of (chosen, rejected) responses used for:
    - Reward model training (learn to score responses)
    - DPO training (directly optimize on preferences)

Dataset: Anthropic/hh-rlhf on Hugging Face
Format: Each example has:
    - "chosen": The preferred response (full conversation)
    - "rejected": The less preferred response (full conversation)

Example from dataset:
    chosen:   "Human: What is 2+2? Assistant: 2+2 equals 4."
    rejected: "Human: What is 2+2? Assistant: The answer is fish."
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class RLHFExample:
    """A single preference example."""
    prompt: str           # The human query
    chosen: str           # The preferred response
    rejected: str         # The less preferred response


class PreferenceDataset(Dataset):
    """Dataset for reward model and DPO training.

    Loads Anthropic HH-RLHF and processes it into (prompt, chosen, rejected) triples.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Step 1: Load the dataset
        #   from datasets import load_dataset
        #   dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        #
        # Step 2: Optionally limit samples (for debugging)
        #   if max_samples: dataset = dataset.select(range(min(max_samples, len(dataset))))
        #
        # Step 3: Parse each example
        #   The raw format has "chosen" and "rejected" as full conversations like:
        #   "\n\nHuman: question\n\nAssistant: answer"
        #
        #   You need to split these into (prompt, response) pairs.
        #   The prompt is everything up to the last "Assistant:" marker.
        #   The response is everything after.
        #
        # Store results in self.examples as a list of RLHFExample
        self.examples: List[RLHFExample] = []
        from datasets import load_dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        for item in dataset:
            chosen_prompt, chosen_response = self._extract_prompt_and_response(item["chosen"])
            rejected_prompt, rejected_response = self._extract_prompt_and_response(item["rejected"])

            # Prompts should usually match; prefer the chosen prompt when available.
            prompt = chosen_prompt if chosen_prompt else rejected_prompt

            if not prompt or not chosen_response or not rejected_response:
                continue

            self.examples.append(
                RLHFExample(
                    prompt=prompt,
                    chosen=chosen_response,
                    rejected=rejected_response,
                )
            )

    def _extract_prompt_and_response(self, text: str) -> Tuple[str, str]:
        """Split a conversation into (prompt, response).

        Input format: "\n\nHuman: What is X?\n\nAssistant: X is Y."
        Output: ("Human: What is X?\n\nAssistant:", "X is Y.")

        For multi-turn, take everything up to the LAST "Assistant:" as prompt.
        """
        
        # Find the last occurrence of "\n\nAssistant:" in text
        # Split there: everything before + "Assistant:" = prompt
        #              everything after = response
        #
        # Handle edge cases:
        #   - Strip whitespace from response
        #   - Handle missing "Assistant:" marker
        
        text = text.strip()
        marker = "\n\nAssistant:"
        if marker not in text:
            # If no marker, treat entire text as prompt and empty response
            return text, ""
        
        last_marker_idx = text.rfind(marker)
        prompt = text[:last_marker_idx + len(marker)].strip()
        response = text[last_marker_idx + len(marker):].strip()

        return prompt, response
        

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized (prompt+chosen, prompt+rejected) pair.

        Returns dict with keys:
            - chosen_input_ids: tokenized prompt + chosen response
            - chosen_attention_mask
            - rejected_input_ids: tokenized prompt + rejected response
            - rejected_attention_mask
            - prompt_length: number of tokens in the prompt (for masking)
        """
        example = self.examples[idx]

        # Step 1: Concatenate prompt + response for both chosen and rejected
        #   chosen_text = example.prompt + " " + example.chosen
        #   rejected_text = example.prompt + " " + example.rejected
        #
        # Step 2: Tokenize both using self.tokenizer
        #   Use: self.tokenizer(text, max_length=self.max_length,
        #                       truncation=True, padding="max_length",
        #                       return_tensors="pt")
        #
        # Step 3: Also tokenize just the prompt to get prompt_length
        #   This is needed later for masking — we only compute loss on
        #   the response tokens, not the prompt.
        #
        # Return a dict with the keys listed above
        chosen_text = f"{example.prompt} {example.chosen}"
        rejected_text = f"{example.prompt} {example.rejected}"

        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prompt_tokens = self.tokenizer(
            example.prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = int(prompt_tokens["attention_mask"].sum().item())

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }


def create_preference_dataloader(
    tokenizer,
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """Convenience function to create a preference DataLoader."""
    dataset = PreferenceDataset(
        tokenizer=tokenizer,
        split=split,
        max_length=max_length,
        max_samples=max_samples,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
