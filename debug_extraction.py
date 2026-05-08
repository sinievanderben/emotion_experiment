#!/usr/bin/env python3
"""
Debug script to verify:
1. Hook firing on distinct modules
2. Output structure of Apertus layers (what is out[0]?)
3. Token aggregation strategy
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Test with Apertus
MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=DEVICE)

# Simple test text
test_text = "This is a test story about emotions. The character felt happy and excited."
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

print(f"\n=== INPUT ===")
print(f"Input shape: {inputs['input_ids'].shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")


print(f"\n=== ISSUE 1: Hook Module Identity ===")

def _get_layer(model: nn.Module, idx: int) -> nn.Module:
    candidates = [
        lambda m, i: m.model.layers[i],
        lambda m, i: m.model.language_model.model.layers[i],
        lambda m, i: m.model.language_model.layers[i],
        lambda m, i: m.language_model.model.layers[i],
    ]
    for fn in candidates:
        try:
            return fn(model, idx)
        except (AttributeError, IndexError):
            continue
    raise AttributeError(f"Cannot locate transformer layer {idx}")

test_layers = [0, 1, 2, 10, 20]
layer_objects = {}
for layer_idx in test_layers:
    layer = _get_layer(model, layer_idx)
    layer_objects[layer_idx] = layer
    print(f"Layer {layer_idx:2d}: {layer}")
    print(f"  id() = {id(layer)}")

# Check if they're distinct
ids = [id(layer_objects[i]) for i in test_layers]
if len(ids) == len(set(ids)):
    print("✓ All layer objects are DISTINCT modules")
else:
    print("✗ WARNING: Some layer objects refer to the SAME module!")


print(f"\n=== ISSUE 2: Output Structure ===")

captured_outputs = {}

def _make_hook(idx: int):
    def _hook(module, inp, out):
        print(f"\nLayer {idx} hook fired:")
        print(f"  type(out) = {type(out)}")
        
        if isinstance(out, tuple):
            print(f"  len(out) = {len(out)}")
            for i, item in enumerate(out[:3]):  # Show first 3 elements
                if hasattr(item, 'shape'):
                    print(f"    out[{i}].shape = {item.shape}, dtype = {item.dtype}")
                else:
                    print(f"    out[{i}] type = {type(item)}")
        else:
            if hasattr(out, 'shape'):
                print(f"  out.shape = {out.shape}, dtype = {out.dtype}")
            else:
                print(f"  out is not a tuple, type = {type(out)}")
        
        # Store what we think is the hidden states
        hs = out[0] if isinstance(out, tuple) else out
        captured_outputs[idx] = {
            'raw_out': out,
            'assumed_hs': hs
        }
    return _hook

hooks = []
for layer_idx in test_layers:
    hooks.append(_get_layer(model, layer_idx).register_forward_hook(_make_hook(layer_idx)))

print("Running forward pass...")
with torch.no_grad():
    outputs = model(**inputs)

for h in hooks:
    h.remove()

# Verify all captured outputs
print(f"\n=== Verification of captured outputs ===")
for layer_idx in test_layers:
    hs = captured_outputs[layer_idx]['assumed_hs']
    print(f"Layer {layer_idx}: assumed_hs.shape = {hs.shape}")
    expected_shape = (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], model.config.hidden_size)
    if hs.shape == expected_shape:
        print(f"  ✓ Shape matches expected (batch, seq, d_model)")
    else:
        print(f"  ✗ Shape mismatch! Expected {expected_shape}, got {hs.shape}")


print(f"\n=== ISSUE 3: Token Aggregation ===")

TOKEN_OFFSET = 50
attention_mask = inputs['attention_mask']
B, S = attention_mask.shape

print(f"Attention mask shape: {attention_mask.shape}")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][:min(10, S)])}")

# Current aggregation strategy
offset_mask = attention_mask.clone()
offset_mask[:, :TOKEN_OFFSET] = 0  # Zero out first TOKEN_OFFSET tokens
om = offset_mask.unsqueeze(-1).float()

real_tokens = int(attention_mask.sum().item())
used_tokens = int(offset_mask.sum().item())

print(f"\nAggregation info:")
print(f"  Real tokens (non-padding): {real_tokens}")
print(f"  Used tokens (after TOKEN_OFFSET={TOKEN_OFFSET} offset): {used_tokens}")
print(f"  Tokens ignored: {real_tokens - used_tokens}")

if used_tokens == 0:
    print(f"  ✗ WARNING: Zero tokens will be aggregated! (sequence too short?)")
else:
    print(f"  ✓ Will aggregate {used_tokens} tokens")

# Show what happens to layer 0 activations
print(f"\n=== Example aggregation (Layer 0) ===")
hidden_0 = captured_outputs[0]['assumed_hs']  # [B, S, D]
B, S, D = hidden_0.shape

agg = (hidden_0 * om).sum(dim=(0, 1))  # Sum over batch and sequence
print(f"Aggregated vector shape: {agg.shape} (should be [{D}])")
print(f"Mean magnitude: {agg.norm().item():.4f}")

# Show per-token magnitudes to understand signal distribution
magnitudes = (hidden_0[0] * om[0]).norm(dim=1)  # [S]
valid_tokens = om[0].squeeze(-1) > 0.5
valid_mags = magnitudes[valid_tokens]
if len(valid_mags) > 0:
    print(f"Token magnitudes (after offset): min={valid_mags.min():.4f}, max={valid_mags.max():.4f}, mean={valid_mags.mean():.4f}")

print("\n=== Done ===")
