#!/usr/bin/env python3
"""
Causal validation of emotion vectors via activation steering.

For each target emotion and each layer:
  1. Project SAE contrast vector → residual-stream direction  (via W_dec)
  2. Hook that layer during generation, add  α × direction  to every token
  3. Generate N completions from a neutral prompt
  4. Score emotional content with the NRC VAD lexicon
  5. Compare steered vs baseline valence/arousal scores

Cross-layer hypothesis test:
  Layers 17-20 (stable geometry) should show stronger steering effect
  than layers 12-16 (pre-transition geometry).

Usage:
    python3 steer_emotion_vectors.py [--layer 18] [--alpha 20] [--n-samples 10]
    python3 steer_emotion_vectors.py --all-layers   # sweep all 6 layers
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")


SCRIPT_DIR      = Path("/users/sinievdben/scratch/personal/emotion_experiment")
VECTORS_DIR     = SCRIPT_DIR / "output/emotion_vectors"
SAE_CONFIG_FILE = SCRIPT_DIR / "sae_config.json"
VAD_CSV         = SCRIPT_DIR / "emotion_valence_arousal_nrc.csv"
OUTPUT_DIR      = SCRIPT_DIR / "output/steering"
MODEL_NAME      = "swiss-ai/Apertus-8B-Instruct-2509"

LAYERS = [12, 16, 17, 18, 19, 20]

# Emotions to steer — chosen for clear valence contrast and NRC coverage
STEER_EMOTIONS = [
    "ecstatic", "joyful", "excited", "grateful",   # high positive valence
    "depressed", "grief-stricken", "gloomy", "furious",  # high negative valence
    "calm", "serene",                               # low arousal
    "panicked", "terrified",                        # high arousal
]

# Neutral prompt — deliberately bland so any emotional signal comes from steering
NEUTRAL_PROMPT = "I sat down and thought about my day. It had been"


def load_vad(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """Load {word: (valence, arousal)} from NRC VAD CSV."""
    vad = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vad[row["emotion"].strip().lower()] = (
                float(row["valence"]), float(row["arousal"])
            )
    return vad


def score_text(text: str, vad: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """
    Score a text by averaging VAD ratings of words that appear in the lexicon.
    Returns (mean_valence, mean_arousal), or (nan, nan) if no words matched.
    """
    words = text.lower().split()
    vals, aros = [], []
    for w in words:
        w = w.strip(".,!?\"'();:-")
        if w in vad:
            v, a = vad[w]
            vals.append(v)
            aros.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.mean(aros))



class BatchTopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k_per_sample: int, device: torch.device):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_in,  d_sae, device=device))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in,  device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_sae,        device=device))
        self.b_dec = nn.Parameter(torch.zeros(d_in,         device=device))
        self.register_buffer("num_batches_not_active",
                             torch.zeros(d_sae, dtype=torch.long, device=device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d_sae] → [..., d_in]"""
        return z @ self.W_dec + self.b_dec


def _resolve_sae_paths(layer: int, sae_cfg: dict) -> Tuple[Path, Path]:
    sae_base = Path(sae_cfg["sae_base"])
    layer_cfg = sae_cfg["layers"][str(layer)]
    run_dir   = sae_base / layer_cfg["run"]
    step      = layer_cfg["step"]
    root      = run_dir if str(step) == "final" else run_dir / f"checkpoint_step_{int(step):06d}"
    return root / "sae_batchtopk_state.pt", root / "sae_config.json"


def load_sae(layer: int, sae_cfg: dict, device: torch.device) -> BatchTopKSAE:
    state_path, config_path = _resolve_sae_paths(layer, sae_cfg)
    with open(config_path) as f:
        cfg = json.load(f)
    sae = BatchTopKSAE(cfg["d_in"], cfg["d_sae"], cfg["k_per_sample"], device)
    sae.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    sae.eval()
    return sae


def compute_steering_direction(
    emotion: str,
    layer: int,
    vectors_dir: Path,
    sae: BatchTopKSAE,
    all_emotions: List[str],
) -> torch.Tensor:
    """
    Returns a unit steering vector in residual-stream space.

    Steps:
      1. Load raw SAE vectors for all emotions at this layer
      2. Subtract cross-emotion mean  (contrast vector)
      3. Select the target emotion's contrast vector
      4. Project through W_dec to get residual-stream direction
      5. Normalise to unit norm
    """
    # Load all emotion vectors
    vecs = []
    for e in all_emotions:
        safe = e.replace(" ", "_").replace("/", "-")
        npy  = vectors_dir / safe / f"layer_{layer}.npy"
        vecs.append(torch.from_numpy(np.load(npy).astype(np.float32)))

    mat  = torch.stack(vecs, dim=0)         # (n_emotions, d_sae)
    mean = mat.mean(dim=0, keepdim=True)
    contrast = mat - mean                   # (n_emotions, d_sae)

    idx   = all_emotions.index(emotion)
    c_vec = contrast[idx].to(sae.W_dec.device)   # (d_sae,)

    # Project to residual-stream space via decoder
    with torch.no_grad():
        direction = sae.decode(c_vec.unsqueeze(0)).squeeze(0)   # (d_in,)

    # Unit norm
    direction = direction / (direction.norm() + 1e-8)
    return direction.cpu()


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    direction: Optional[torch.Tensor],   # None → baseline (no steering)
    alpha: float,
    n_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate text, optionally adding  alpha * direction  to residual stream at `layer`."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    hooks = []
    if direction is not None:
        dir_dev = direction.to(device)

        def _hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs + alpha * dir_dev.view(1, 1, -1).to(hs.dtype)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs

        hooks.append(model.model.layers[layer].register_forward_hook(_hook))

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=n_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    for h in hooks:
        h.remove()

    output_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def run_steering_experiment(
    emotions: List[str],
    layers: List[int],
    alpha: float,
    n_samples: int,
    n_new_tokens: int,
    vectors_dir: Path,
    sae_cfg: dict,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    vad = load_vad(VAD_CSV)

    # Load all emotion names (for contrast computation)
    all_emotions = sorted([
        d.name for d in vectors_dir.iterdir() if d.is_dir()
    ])
    # Map to canonical names (underscore → space)
    all_emotions_canon = [e.replace("_", " ") for e in all_emotions]

    print(f"Loading model {MODEL_NAME} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    results = []   # list of dicts, one per (emotion, layer, sample)

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        sae = load_sae(layer, sae_cfg, device)

        for emotion in emotions:
            # Find canonical dir name
            safe = emotion.replace(" ", "_").replace("/", "-")
            if safe not in all_emotions:
                print(f"  [{emotion}] not found in vectors_dir — skipping")
                continue

            # Resolve canonical list index
            emo_idx_safe = all_emotions.index(safe)
            all_emotions_for_contrast = [e.replace("_", " ") for e in all_emotions]

            direction = compute_steering_direction(
                emotion=all_emotions_for_contrast[emo_idx_safe],
                layer=layer,
                vectors_dir=vectors_dir,
                sae=sae,
                all_emotions=all_emotions_for_contrast,
            ).to(device)

            print(f"\n  [{emotion}]")

            for sample_i in range(n_samples):
                # Baseline
                baseline_text = generate_with_steering(
                    model, tokenizer, NEUTRAL_PROMPT,
                    layer=layer, direction=None, alpha=alpha,
                    n_new_tokens=n_new_tokens, device=device,
                )
                base_val, base_aro = score_text(baseline_text, vad)

                # Steered
                steered_text = generate_with_steering(
                    model, tokenizer, NEUTRAL_PROMPT,
                    layer=layer, direction=direction, alpha=alpha,
                    n_new_tokens=n_new_tokens, device=device,
                )
                steer_val, steer_aro = score_text(steered_text, vad)

                results.append({
                    "emotion":        emotion,
                    "layer":          layer,
                    "sample":         sample_i,
                    "alpha":          alpha,
                    "baseline_val":   base_val,
                    "baseline_aro":   base_aro,
                    "steered_val":    steer_val,
                    "steered_aro":    steer_aro,
                    "delta_val":      steer_val - base_val,
                    "delta_aro":      steer_aro - base_aro,
                    "baseline_text":  baseline_text,
                    "steered_text":   steered_text,
                })

                print(f"    sample {sample_i}: Δval={steer_val - base_val:+.3f}  "
                      f"Δaro={steer_aro - base_aro:+.3f}")

        # Unload SAE to free memory between layers
        del sae
        torch.cuda.empty_cache()

    results_path = output_dir / "steering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results → {results_path}")

    summary = {}
    for r in results:
        key = (r["emotion"], r["layer"])
        if key not in summary:
            summary[key] = {"delta_vals": [], "delta_aros": []}
        if not np.isnan(r["delta_val"]):
            summary[key]["delta_vals"].append(r["delta_val"])
        if not np.isnan(r["delta_aro"]):
            summary[key]["delta_aros"].append(r["delta_aro"])

    print("\n" + "="*60)
    print("STEERING SUMMARY  (mean Δvalence per layer)")
    print("="*60)
    print(f"{'Emotion':<20} " + "  ".join(f"L{l:>2}" for l in layers))
    print("-"*60)
    for emotion in emotions:
        row = f"{emotion:<20}"
        for layer in layers:
            key = (emotion, layer)
            if key in summary and summary[key]["delta_vals"]:
                row += f"  {np.mean(summary[key]['delta_vals']):+.2f}"
            else:
                row += "    -- "
        print(row)

    print("\nMean |Δvalence| per layer (effect size):")
    for layer in layers:
        delta_vals = [
            v for (e, l), d in summary.items()
            if l == layer for v in d["delta_vals"]
        ]
        if delta_vals:
            print(f"  Layer {layer}: {np.mean(np.abs(delta_vals)):.3f} "
                  f"(n={len(delta_vals)})")

    summary_path = output_dir / "steering_summary.json"
    agg = {
        f"{e}__L{l}": {
            "mean_delta_val": float(np.mean(d["delta_vals"])) if d["delta_vals"] else None,
            "mean_delta_aro": float(np.mean(d["delta_aros"])) if d["delta_aros"] else None,
            "n": len(d["delta_vals"]),
        }
        for (e, l), d in summary.items()
    }
    with open(summary_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Summary → {summary_path}")


def plot_results(results_path: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    with open(results_path) as f:
        results = json.load(f)

    layers   = sorted(set(r["layer"]   for r in results))
    emotions = sorted(set(r["emotion"] for r in results))

    layer_effect = {}
    for layer in layers:
        deltas = [r["delta_val"] for r in results
                  if r["layer"] == layer and not np.isnan(r["delta_val"])]
        layer_effect[layer] = float(np.mean(np.abs(deltas))) if deltas else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    xs = list(layer_effect.keys())
    ys = [layer_effect[l] for l in xs]
    colors = ["#78909C" if l < 17 else "#1976D2" for l in xs]
    bars = ax.bar([str(l) for l in xs], ys, color=colors, edgecolor="k", linewidth=0.5)
    ax.axvline(1.5, color="red", lw=1, ls="--", label="phase transition")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |Δvalence|")
    ax.set_title("Steering effect size per layer\n(higher = stronger causal effect)")
    ax.legend(fontsize=8)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.002, f"{y:.3f}",
                ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    mat = np.full((len(emotions), len(layers)), np.nan)
    for i, e in enumerate(emotions):
        for j, l in enumerate(layers):
            deltas = [r["delta_val"] for r in results
                      if r["emotion"] == e and r["layer"] == l
                      and not np.isnan(r["delta_val"])]
            if deltas:
                mat[i, j] = np.mean(deltas)

    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="mean Δvalence")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title("Δvalence per emotion × layer\n(red = more positive, blue = more negative)")
    ax.axvline(1.5, color="red", lw=1.5, ls="--")

    fig.suptitle("Causal validation: activation steering (Apertus 8B)", y=1.02)
    fig.tight_layout()
    out = output_dir / "fig5_steering.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir",  type=Path, default=VECTORS_DIR)
    parser.add_argument("--sae-config",   type=Path, default=SAE_CONFIG_FILE)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--layer",        type=int,  default=None,
                        help="Single layer to steer (default: all)")
    parser.add_argument("--all-layers",   action="store_true",
                        help="Sweep all layers (overrides --layer)")
    parser.add_argument("--alpha",        type=float, default=20.0,
                        help="Steering strength")
    parser.add_argument("--n-samples",    type=int,  default=10,
                        help="Generations per (emotion, layer) pair")
    parser.add_argument("--n-new-tokens", type=int,  default=80)
    parser.add_argument("--plot-only",    type=Path,  default=None,
                        help="Skip generation, just plot existing results JSON")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.plot_only, args.output_dir)
        return

    if args.all_layers:
        layers = LAYERS
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = LAYERS

    with open(args.sae_config) as f:
        sae_cfg = json.load(f)

    run_steering_experiment(
        emotions=STEER_EMOTIONS,
        layers=layers,
        alpha=args.alpha,
        n_samples=args.n_samples,
        n_new_tokens=args.n_new_tokens,
        vectors_dir=args.vectors_dir,
        sae_cfg=sae_cfg,
        output_dir=args.output_dir,
        device=torch.device(args.device),
    )

    plot_results(args.output_dir / "steering_results.json", args.output_dir)


if __name__ == "__main__":
    main()
