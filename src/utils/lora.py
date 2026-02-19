"""LoRA (Low-Rank Adaptation) utilities.

This is the FIRST file you should implement. Understanding LoRA is foundational
to everything else in this project.

Key paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

Core idea:
    Instead of updating a full weight matrix W (d x d), we learn two small matrices:
    W' = W + (alpha/r) * B @ A
    Where A is (r x d) and B is (d x r), with r << d

    This means instead of training d*d parameters, we only train 2*d*r parameters.
    For r=16, d=4096 (Llama-7B hidden size): 16M → 131K per layer. ~120x reduction!

Implementation order:
    1. LoRALinear (the core building block)
    2. find_target_modules (which layers to apply LoRA to)
    3. apply_lora_to_model (wrap existing layers)
    4. get_lora_parameters (for optimizer)
    5. merge_lora_weights (for inference)
"""
import math
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A Linear layer augmented with LoRA adapters.

    This wraps an existing nn.Linear layer and adds low-rank adaptation.
    The original weight is FROZEN. Only A and B matrices are trainable.

    Forward pass: y = x @ W^T + (alpha/r) * x @ A^T @ B^T

    Args:
        base_layer: The original nn.Linear layer (will be frozen)
        r: Rank of the low-rank decomposition (typical: 8, 16, 32, 64)
        alpha: Scaling factor (typical: 16, 32). Higher = stronger adaptation
        dropout: Dropout applied to LoRA path
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze the original weight
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Create two nn.Parameter tensors:
        #   self.lora_A: shape (r, in_features)
        #   self.lora_B: shape (out_features, r)
        #
        # Initialization matters!
        #   - A: Use Kaiming uniform (nn.init.kaiming_uniform_ with a=math.sqrt(5))
        #   - B: Initialize to ZEROS
        #
        # Why zeros for B? So that at the start of training, B @ A = 0,
        # meaning the model behaves exactly like the original pretrained model.
        # Training gradually learns the adaptation from this "zero init" state.

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout on the LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: base output + scaled LoRA output.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Step 1: Compute the base layer output (frozen weights)
        #   base_output = self.base_layer(x)
        #
        # Step 2: Compute the LoRA path
        #   - Apply dropout to x
        #   - Multiply: x @ A^T → (batch, r)
        #   - Multiply: result @ B^T → (batch, out_features)
        #   - Scale by self.scaling
        #
        # Step 3: Add them together
        #   return base_output + lora_output
        #
        # Note: Be careful with tensor shapes. x might be 3D (batch, seq, features).
        # F.linear(x, weight) computes x @ weight^T, which handles batched input.
        # ============================================================
        base_output = self.base_layer(x)
        lora_input = self.lora_dropout(x)
        lora_output = F.linear(lora_input, self.lora_A)  # (batch, r)
        lora_output = F.linear(lora_output, self.lora_B)  # (batch, out_features)
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base layer for inference.

        Returns a plain nn.Linear with W_merged = W + (alpha/r) * B @ A
        This eliminates the LoRA overhead during inference.
        """
        # ============================================================
        # TODO: Merge LoRA weights back into base layer
        # ============================================================
        #
        # Step 1: Compute delta_W = self.scaling * (self.lora_B @ self.lora_A)
        # Step 2: Add delta_W to self.base_layer.weight.data
        # Step 3: Return self.base_layer
        #
        # After merging, the layer behaves identically but without
        # the extra A, B matrix multiplications.
        # ============================================================
        raise NotImplementedError("Implement LoRA weight merging")


def find_target_modules(model: nn.Module, target_names: List[str]) -> List[str]:
    """Find all modules in the model that match target names.

    For Llama, the typical targets are:
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    For a minimal setup (less memory), just target attention:
        ["q_proj", "v_proj"]

    Args:
        model: The pretrained model
        target_names: List of module name suffixes to match

    Returns:
        List of full module paths (e.g., "model.layers.0.self_attn.q_proj")
    """
    # Iterate over model.named_modules()
    # For each (name, module):
    #   - Check if module is nn.Linear
    #   - Check if the last part of `name` (after the last ".") is in target_names
    #   - If both true, add `name` to the result list
    #
    # Return the list of matching module paths
    # ============================================================
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in target_names:
            print(f"Found target module: {name}")
            target_modules.append(name)
    return target_modules


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = None,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Apply LoRA adapters to specified modules in the model.

    This function:
    1. Freezes ALL model parameters
    2. Replaces target Linear layers with LoRALinear
    3. Only LoRA parameters (A, B matrices) are trainable

    Args:
        model: The pretrained model
        target_modules: Which modules to apply LoRA to (default: attention projections)
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: LoRA dropout

    Returns:
        The modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Step 1: Freeze all parameters
    #   for param in model.parameters():
    #       param.requires_grad = False
    #
    # Step 2: Find target modules using find_target_modules()
    #
    # Step 3: For each target module path, replace it with LoRALinear
    #   This is tricky! You need to:
    #   a) Split the path by "." to navigate the module hierarchy
    #   b) Get the parent module
    #   c) Get the original Linear layer
    #   d) Create a LoRALinear wrapping it
    #   e) Use setattr(parent, child_name, lora_layer) to replace it
    #
    # Step 4: Print summary (how many LoRA params vs total params)
    for param in model.parameters():
        param.requires_grad = False

    target_modules = find_target_modules(model, target_modules)
    for name in target_modules:
        parts = name.split(".")
        # get parent
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        original_layer = getattr(parent, child_name)
        lora_layer = LoRALinear(original_layer, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, lora_layer)
    # Print summary of trainable parameters
    print_trainable_parameters(model)
    return model

def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get only the trainable LoRA parameters.

    Used to create an optimizer that only updates LoRA weights.

    Returns:
        List of trainable parameters (only lora_A and lora_B)
    """
    # ============================================================
    # TODO: Collect all trainable parameters
    # ============================================================
    #
    # Simple approach: return [p for p in model.parameters() if p.requires_grad]
    #
    # Better approach: explicitly find LoRALinear modules and get their A, B
    # ============================================================
    raise NotImplementedError("Implement LoRA parameter collection")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA weights back into the base model for inference.

    After merging, the model has no LoRA overhead but produces identical outputs.

    Returns:
        Model with merged weights (plain Linear layers, no LoRA)
    """
    # ============================================================
    # TODO: Find all LoRALinear modules and call .merge() on them
    # ============================================================
    #
    # For each named_module in the model:
    #   If it's a LoRALinear instance, replace it with merged result
    #   (Same parent/child navigation as apply_lora_to_model)
    # ============================================================
    raise NotImplementedError("Implement LoRA weight merging for full model")


def print_trainable_parameters(model: nn.Module) -> dict:
    """Print and return trainable parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")
    print(f"Trainable %:          {100 * trainable_params / total_params:>11.2f}%")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "percentage": 100 * trainable_params / total_params,
    }
