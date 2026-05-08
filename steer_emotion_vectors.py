#!/usr/bin/env python3
"""
Small activation-steering experiment using pre-extracted emotion vectors.

For each (emotion, layer):
  1. Load residual-stream emotion vectors, mean-centre → contrast vector
  2. Hook that layer during generation, add α × unit_direction to every token
  3. Generate N completions from a neutral prompt
  4. Score emotional content with the NRC VAD lexicon
  5. Compare steered vs baseline valence/arousal

Usage:
    python3 steer_emotion_vectors.py \
        --model-name  swiss-ai/Apertus-8B-Instruct-2509 \
        --vectors-dir output_apertus/emotion_vectors \
        --layers      16 20 24 28 \
        --emotions    "joyful" "depressed" "calm" "panicked" \
        --alpha       20 \
        --n-samples   5 \
        --output-dir  output_steering/apertus_steering
"""

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
VAD_CSV    = SCRIPT_DIR / "emotion_valence_arousal_nrc.csv"

NEUTRAL_PROMPT = "I sat down and thought about my day. It had been"


def load_vad(csv_path: Path) -> dict[str, tuple[float, float]]:
    vad = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vad[row["emotion"].strip().lower()] = (
                float(row["valence"]), float(row["arousal"])
            )
    return vad


def score_text(text: str, vad: dict) -> tuple[float, float]:
    words = text.lower().split()
    vals, aros = [], []
    for w in words:
        w = w.strip(".,!?\"'();:-")
        if w in vad:
            vals.append(vad[w][0])
            aros.append(vad[w][1])
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.mean(aros))


def _layer_file(emo_dir: Path, layer: int) -> Optional[Path]:
    for name in (f"layer_{layer}_resid_projected.npy", f"layer_{layer}_resid.npy"):
        p = emo_dir / name
        if p.exists():
            return p
    return None


def compute_steering_direction(
    emotion: str,
    layer: int,
    vectors_dir: Path,
    all_emotions: list[str],
) -> torch.Tensor:
    """
    Load residual-stream vectors for all emotions, mean-centre, return the
    unit-norm contrast vector for `emotion`.
    """
    vecs = []
    for e in all_emotions:
        p = _layer_file(vectors_dir / e, layer)
        if p is None:
            raise FileNotFoundError(f"No layer-{layer} file for emotion '{e}' in {vectors_dir}")
        vecs.append(torch.from_numpy(np.load(p).astype(np.float32)))

    mat      = torch.stack(vecs, dim=0)          # (n_emotions, d_model)
    contrast = mat - mat.mean(dim=0, keepdim=True)
    direction = contrast[all_emotions.index(emotion)]
    return direction / (direction.norm() + 1e-8)


def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    direction: Optional[torch.Tensor],
    alpha: float,
    n_new_tokens: int,
    device: torch.device,
) -> str:
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
            return (hs,) + out[1:] if isinstance(out, tuple) else hs

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
    model_name: str,
    emotions: list[str],
    layers: list[int],
    alpha: float,
    n_samples: int,
    n_new_tokens: int,
    vectors_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vad = load_vad(VAD_CSV)

    all_emotions = sorted([
        d.name for d in vectors_dir.iterdir()
        if d.is_dir() and _layer_file(d, layers[0]) is not None
    ])
    missing = [e for e in emotions if e not in all_emotions]
    if missing:
        raise ValueError(f"Emotions not found in {vectors_dir}: {missing}")

    print(f"Loading {model_name} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    results = []

    for layer in layers:
        print(f"\n{'='*60}\nLayer {layer}\n{'='*60}")
        for emotion in emotions:
            direction = compute_steering_direction(
                emotion, layer, vectors_dir, all_emotions
            ).to(device)

            print(f"\n  [{emotion}]")
            for sample_i in range(n_samples):
                base_text  = generate_with_steering(
                    model, tokenizer, NEUTRAL_PROMPT,
                    layer=layer, direction=None, alpha=alpha,
                    n_new_tokens=n_new_tokens, device=device,
                )
                steer_text = generate_with_steering(
                    model, tokenizer, NEUTRAL_PROMPT,
                    layer=layer, direction=direction, alpha=alpha,
                    n_new_tokens=n_new_tokens, device=device,
                )
                base_val,  base_aro  = score_text(base_text,  vad)
                steer_val, steer_aro = score_text(steer_text, vad)

                results.append({
                    "emotion": emotion, "layer": layer, "sample": sample_i,
                    "alpha": alpha,
                    "baseline_val": base_val,  "baseline_aro": base_aro,
                    "steered_val":  steer_val, "steered_aro":  steer_aro,
                    "delta_val":    steer_val - base_val,
                    "delta_aro":    steer_aro - base_aro,
                    "baseline_text": base_text,
                    "steered_text":  steer_text,
                })
                print(f"    sample {sample_i}: Δval={steer_val-base_val:+.3f}  "
                      f"Δaro={steer_aro-base_aro:+.3f}")

    results_path = output_dir / "steering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results → {results_path}")

    _print_summary(results, emotions, layers)
    _save_summary(results, emotions, layers, output_dir)


def _print_summary(results, emotions, layers):
    print("\n" + "="*60)
    print("SUMMARY  (mean Δvalence per layer)")
    print("="*60)
    print(f"{'Emotion':<20} " + "  ".join(f"L{l:>2}" for l in layers))
    print("-"*60)
    for emotion in emotions:
        row = f"{emotion:<20}"
        for layer in layers:
            deltas = [r["delta_val"] for r in results
                      if r["emotion"] == emotion and r["layer"] == layer
                      and not np.isnan(r["delta_val"])]
            row += f"  {np.mean(deltas):+.2f}" if deltas else "    -- "
        print(row)


def _save_summary(results, emotions, layers, output_dir):
    agg = {}
    for emotion in emotions:
        for layer in layers:
            deltas_v = [r["delta_val"] for r in results
                        if r["emotion"] == emotion and r["layer"] == layer
                        and not np.isnan(r["delta_val"])]
            deltas_a = [r["delta_aro"] for r in results
                        if r["emotion"] == emotion and r["layer"] == layer
                        and not np.isnan(r["delta_aro"])]
            agg[f"{emotion}__L{layer}"] = {
                "mean_delta_val": float(np.mean(deltas_v)) if deltas_v else None,
                "mean_delta_aro": float(np.mean(deltas_a)) if deltas_a else None,
                "n": len(deltas_v),
            }
    with open(output_dir / "steering_summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(f"Summary → {output_dir / 'steering_summary.json'}")


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

    # Figure 1: mean |Δvalence| per layer
    fig, ax = plt.subplots(figsize=(8, 5))
    layer_effect = {}
    for layer in layers:
        deltas = [r["delta_val"] for r in results
                  if r["layer"] == layer and not np.isnan(r["delta_val"])]
        layer_effect[layer] = float(np.mean(np.abs(deltas))) if deltas else 0.0
    bars = ax.bar([str(l) for l in layers],
                  [layer_effect[l] for l in layers],
                  color="#1976D2", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |Δvalence|")
    ax.set_title("Steering effect size per layer")
    for bar, y in zip(bars, [layer_effect[l] for l in layers]):
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.002, f"{y:.3f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out1 = output_dir / "fig_steering_layer_effect.pdf"
    fig.savefig(out1)
    plt.close(fig)
    print(f"  saved {out1.name}")

    # Figure 2: heatmap Δvalence per (emotion × layer)
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = np.full((len(emotions), len(layers)), np.nan)
    for i, e in enumerate(emotions):
        for j, l in enumerate(layers):
            deltas = [r["delta_val"] for r in results
                      if r["emotion"] == e and r["layer"] == l
                      and not np.isnan(r["delta_val"])]
            if deltas:
                mat[i, j] = np.mean(deltas)
    vmax = max(np.nanmax(np.abs(mat)), 1e-6)
    im = ax.imshow(mat, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.04, label="mean Δvalence")
    ax.set_xticks(range(len(layers)));   ax.set_xticklabels([str(l) for l in layers])
    ax.set_yticks(range(len(emotions))); ax.set_yticklabels(emotions, fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_title("Δvalence per emotion × layer")
    fig.tight_layout()
    out2 = output_dir / "fig_steering_heatmap.pdf"
    fig.savefig(out2)
    plt.close(fig)
    print(f"  saved {out2.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",   type=str, required=True)
    parser.add_argument("--vectors-dir",  type=Path, required=True)
    parser.add_argument("--layers",       type=int, nargs="+", required=True)
    parser.add_argument("--emotions",     type=str, nargs="+",
                        default=["joyful", "depressed", "calm", "panicked",
                                 "excited", "gloomy"])
    parser.add_argument("--alpha",        type=float, default=20.0)
    parser.add_argument("--n-samples",    type=int,   default=5)
    parser.add_argument("--n-new-tokens", type=int,   default=80)
    parser.add_argument("--output-dir",   type=Path,  required=True)
    parser.add_argument("--plot-only",    type=Path,  default=None,
                        help="Skip generation; plot an existing results JSON")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.plot_only:
        plot_results(args.plot_only, args.output_dir)
        return

    run_steering_experiment(
        model_name=args.model_name,
        emotions=args.emotions,
        layers=args.layers,
        alpha=args.alpha,
        n_samples=args.n_samples,
        n_new_tokens=args.n_new_tokens,
        vectors_dir=args.vectors_dir,
        output_dir=args.output_dir,
        device=torch.device(args.device),
    )
    plot_results(args.output_dir / "steering_results.json", args.output_dir)


if __name__ == "__main__":
    main()
