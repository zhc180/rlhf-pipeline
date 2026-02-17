# RLHF Pipeline — Implementation Guide

## Overview

You'll implement 18 functions across 7 files, in a carefully ordered sequence
where each function builds on what you've already done.

**Total estimated time: 2 weeks (Weeks 3-4 of your 8-week plan)**

---

## Implementation Order

```
Phase 1: LoRA (Day 1-2)            ← Foundation for everything
    │
Phase 2: Data (Day 3)              ← Load preference pairs
    │
Phase 3: Reward Model (Day 4-5)    ← Learn to score responses
    │
    ├── Phase 4: DPO (Day 6-8)     ← Simpler alignment method
    │
    └── Phase 5: PPO (Day 9-12)    ← Full RL-based alignment
          │
Phase 6: Evaluation (Day 13-14)    ← Compare everything
```

---

## Phase 1: LoRA (`src/utils/lora.py`) — Days 1-2

LoRA is the foundation. Every other module depends on it.

### Function Order:

#### 1.1 `LoRALinear.__init__` (30 min)
**What:** Initialize the A and B low-rank matrices.
**Test:** Create a LoRALinear, print parameter counts.
```python
base = nn.Linear(4096, 4096)
lora = LoRALinear(base, r=16)
# A should be (16, 4096), B should be (4096, 16)
# B should be all zeros
```
**Read before:** LoRA paper Section 4.1

#### 1.2 `LoRALinear.forward` (30 min)
**What:** Forward pass: base output + scaled LoRA output.
**Test:** Output shape should match base layer. With zero-init B, output should equal base output.
```python
x = torch.randn(2, 128, 4096)
assert torch.allclose(lora(x), base(x), atol=1e-5)  # B is zeros!
```

#### 1.3 `find_target_modules` (20 min)
**What:** Walk model tree, find Linear layers matching target names.
**Test:** Load a small model, verify it finds the right layers.
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
modules = find_target_modules(model, ["c_attn", "c_proj"])
print(modules)  # Should list all attention projections
```

#### 1.4 `apply_lora_to_model` (45 min)
**What:** Replace target Linear layers with LoRALinear.
**Tricky part:** Navigating the nested module hierarchy with `getattr`/`setattr`.
**Test:** After applying, only LoRA params should be trainable.
```python
apply_lora_to_model(model, target_modules=["c_attn"])
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/total:.2%}")  # Should be ~1-3%
```

#### 1.5 `get_lora_parameters` (10 min)
**What:** Collect all trainable parameters for the optimizer.
**Test:** Length should equal number of LoRA layers × 2 (A + B per layer).

#### 1.6 `merge_lora_weights` (20 min)
**What:** Merge A/B back into base weights for inference.
**Test:** Output should be identical before/after merge, but forward should be faster.

**✅ Phase 1 checkpoint:** You should be able to:
```python
model = AutoModelForCausalLM.from_pretrained("gpt2")
apply_lora_to_model(model, r=16)
print_trainable_parameters(model)  # ~1% trainable
output1 = model(dummy_input)
merge_lora_weights(model)
output2 = model(dummy_input)
# output1 ≈ output2 (numerical precision)
```

---

## Phase 2: Data (`src/utils/data.py`) — Day 3

### Function Order:

#### 2.1 `PreferenceDataset._extract_prompt_and_response` (20 min)
**What:** Parse Anthropic's conversation format.
**Test:**
```python
text = "\n\nHuman: What is 2+2?\n\nAssistant: The answer is 4."
prompt, response = _extract_prompt_and_response(text)
assert "Human: What is 2+2?" in prompt
assert response == "The answer is 4."
```

#### 2.2 `PreferenceDataset.__init__` (30 min)
**What:** Load HH-RLHF dataset and parse into examples.
**Test:**
```python
dataset = PreferenceDataset(tokenizer, split="train", max_samples=100)
assert len(dataset) == 100
assert dataset.examples[0].prompt  # Non-empty
assert dataset.examples[0].chosen != dataset.examples[0].rejected
```

#### 2.3 `PreferenceDataset.__getitem__` (30 min)
**What:** Tokenize a preference pair.
**Test:** Check shapes, verify padding, check prompt_length is correct.

**✅ Phase 2 checkpoint:** You should be able to iterate through a DataLoader
and see batches of tokenized preference pairs.

---

## Phase 3: Reward Model — Days 4-5

### 3.1 `RewardModel.__init__` (`src/reward_model/model.py`) (30 min)
**What:** Load LLM backbone, apply LoRA, add reward head.
**Tip:** Start with a smaller model (GPT-2) for debugging, then switch to Llama-7B.

### 3.2 `RewardModel.forward` (`src/reward_model/model.py`) (30 min)
**What:** Forward pass → extract last token hidden state → project to scalar.
**Critical detail:** Handle padding correctly when finding the "last" token.
**Test:**
```python
rm = RewardModel("gpt2")  # Debug with small model first
input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
reward = rm(input_ids)
assert reward.shape == (1,)  # Single scalar
```

### 3.3 `compute_preference_loss` (`src/reward_model/model.py`) (15 min)
**What:** Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected)).
**Test:** If chosen_rewards > rejected_rewards, loss should be small.
```python
chosen = torch.tensor([2.0, 1.5])
rejected = torch.tensor([0.5, 0.3])
loss, metrics = compute_preference_loss(chosen, rejected)
assert metrics["accuracy"] == 1.0  # Both chosen > rejected
```

### 3.4 `RewardModelTrainer.__init__` (`src/reward_model/trainer.py`) (20 min)
### 3.5 `RewardModelTrainer.train_step` (30 min)
### 3.6 `RewardModelTrainer.evaluate` (20 min)
### 3.7 `RewardModelTrainer.train` (30 min)
### 3.8 `RewardModelTrainer.save` (15 min)

**✅ Phase 3 checkpoint:** Train a reward model for 100 steps. Accuracy should climb
from ~50% (random) toward 65%+ within the first few hundred steps.
If accuracy doesn't improve, check: loss is decreasing, gradients are flowing
(only through LoRA params), learning rate isn't too high/low.

---

## Phase 4: DPO (`src/dpo/trainer.py`) — Days 6-8

### 4.1 `DPOTrainer.__init__` (30 min)
**What:** Load policy model + reference model. Apply LoRA to policy only.
**Memory tip:** Two 7B models in BF16 = ~28 GB. Fits on an A100-40GB.

### 4.2 `DPOTrainer.compute_log_probs` (45 min) ⭐ KEY FUNCTION
**What:** Compute sum of log probs for response tokens only.
**This is the hardest function in Phase 4.** Get this right and everything else follows.
**Test:**
```python
# Log probs should be negative (probabilities < 1)
logps = compute_log_probs(model, input_ids, attention_mask, prompt_lengths)
assert (logps < 0).all()

# Longer sequences → more negative log probs (more tokens = lower joint prob)
```

### 4.3 `DPOTrainer.compute_dpo_loss` (20 min)
**What:** The DPO loss formula.
**Test:** If policy strongly prefers chosen (and ref doesn't), loss should be low.

### 4.4 `DPOTrainer.train_step` (20 min)
### 4.5 `DPOTrainer.train` (20 min)

**✅ Phase 4 checkpoint:** DPO training should show:
- Loss decreasing from ~0.69 (log(2), random) toward ~0.4-0.5
- Accuracy (implicit rewards) climbing above 60%
- Chosen log-ratio increasing, rejected decreasing

---

## Phase 5: PPO (`src/ppo/trainer.py`) — Days 9-12

This is the most complex phase. Take your time.

### 5.1 `ValueHead.__init__` + `forward` (15 min)
**What:** Small MLP: hidden_size → hidden_size → 1.

### 5.2 `PPOTrainer.__init__` (30 min)
**What:** Initialize 4 models (policy, ref, reward, value head).

### 5.3 `PPOTrainer.generate_responses` (20 min)
**What:** Use policy model to generate text responses.

### 5.4 `PPOTrainer.compute_rewards_and_values` (45 min)
**What:** Score generations + compute KL penalties.
**Critical:** KL penalty prevents reward hacking (model exploiting reward model).

### 5.5 `PPOTrainer.compute_advantages` (30 min) ⭐ KEY FUNCTION
**What:** GAE (Generalized Advantage Estimation).
**Read before:** PPO paper Section 3, Spinning Up GAE explanation.
**Debug tip:** Print advantages — they should be centered around 0 and have
reasonable magnitudes (not exploding).

### 5.6 `PPOTrainer.ppo_step` (45 min) ⭐ KEY FUNCTION
**What:** The PPO clipped objective.
**Critical details:**
- Ratio should be close to 1.0 initially (same policy generated the data)
- Clip fraction tells you how often the clipping is active
- If approx_kl > 0.02, stop the inner epoch early

### 5.7 `PPOTrainer.train_step` (30 min)
**What:** Orchestrate: generate → score → GAE → PPO updates.

### 5.8 `PPOTrainer.train` (20 min)

**✅ Phase 5 checkpoint:** PPO training should show:
- Reward gradually increasing
- KL divergence staying bounded (not exploding)
- Clip fraction around 10-30% (if 0%, clip_range is too wide)
- Generated text becoming more helpful/aligned

---

## Phase 6: Evaluation (`src/evaluation/evaluate.py`) — Days 13-14

### 6.1 Generate responses from: base model, DPO model, PPO model
### 6.2 Score all with reward model
### 6.3 Compute diversity metrics
### 6.4 Create comparison report

**✅ Final checkpoint:** A markdown report showing:
- PPO and DPO both improve over base model
- Quantitative comparison (reward scores, diversity)
- Qualitative examples (side-by-side responses)

---

## Debugging Cheat Sheet

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss doesn't decrease | LoRA params not getting gradients | Check `requires_grad`, check optimizer param groups |
| NaN loss | Learning rate too high | Lower LR, check for div-by-zero in log_probs |
| Reward accuracy stuck at 50% | Not learning from preferences | Verify data parsing, check chosen ≠ rejected |
| DPO loss stuck at 0.693 | Log probs not computed correctly | Debug `compute_log_probs` with known inputs |
| PPO reward goes up then crashes | Reward hacking | Increase `kl_coeff` |
| PPO KL explodes | LR too high or clip_range too wide | Lower LR, reduce clip_range to 0.1 |
| OOM on Llama-7B | Too much memory | Reduce batch size, use gradient accumulation |

---

## Recommended Reading Order

Before each phase, read:

| Phase | Read This |
|-------|-----------|
| 1 (LoRA) | [LoRA paper](https://arxiv.org/abs/2106.09685) Sections 1, 3, 4.1 |
| 2 (Data) | Browse the [HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) on HF |
| 3 (Reward) | [InstructGPT paper](https://arxiv.org/abs/2203.02155) Section 3.2 (Reward Model) |
| 4 (DPO) | [DPO paper](https://arxiv.org/abs/2305.18290) Sections 1-4 |
| 5 (PPO) | [PPO paper](https://arxiv.org/abs/1707.06347) + [Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) |
| 6 (Eval) | Skim the TRL library's evaluation utilities for inspiration |
