# RLHF Pipeline: PPO vs DPO with LoRA

End-to-end Reinforcement Learning from Human Feedback pipeline,
comparing PPO and DPO for aligning Llama-2-7B using LoRA fine-tuning.

## Architecture

```
                    Preference Data (Anthropic HH-RLHF)
                         ┌──────┴──────┐
                         │             │
                   ┌─────▼─────┐ ┌────▼─────┐
                   │  Reward   │ │   DPO    │
                   │  Model    │ │ Trainer  │
                   │ (LoRA)    │ │ (LoRA)   │
                   └─────┬─────┘ └──────────┘
                         │
                   ┌─────▼─────┐
                   │   PPO     │
                   │  Trainer  │
                   │ (LoRA)    │
                   └───────────┘
                         │
                   ┌─────▼─────┐
                   │ Evaluate  │
                   │ PPO vs DPO│
                   └───────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# 1. Train reward model
python -m src.reward_model.trainer

# 2. Train with DPO
python -m src.dpo.trainer

# 3. Train with PPO (requires trained reward model)
python -m src.ppo.trainer

# 4. Evaluate and compare
python -m src.evaluation.evaluate
```

## Implementation Guide

See [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) for a detailed,
function-by-function implementation walkthrough.

## Key Concepts

- **LoRA**: Low-rank adaptation — train ~1% of parameters, same quality
- **Reward Model**: Learns to score responses based on human preferences
- **DPO**: Directly optimize on preferences (no RL, simpler)
- **PPO**: RL-based optimization (more complex, but supports online learning)

## Results

<!-- Fill in after training -->

| Method | Avg Reward | Accuracy | Notes |
|--------|-----------|----------|-------|
| Base   |           |          |       |
| DPO    |           |          |       |
| PPO    |           |          |       |

## Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Anthropic HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
