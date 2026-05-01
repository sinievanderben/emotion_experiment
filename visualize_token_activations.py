#!/usr/bin/env python3
"""
Batch token-level emotion vector activation visualiser.

For each sentence in sentences.json:
  1. Run a forward pass and capture per-token residual-stream activations at
     the target layer via a forward hook.
  2. Project each token's activation onto unit-norm emotion contrast vectors.
  3. Produce two figures per sentence:
       <label>_token_heatmap.pdf  — tokens coloured by projection onto the
                                    primary emotion's contrast vector
       <label>_emotion_bar.pdf    — mean token projection for a set of emotions,
                                    showing primary > related > unrelated

Usage:
    python3 visualize_token_activations.py \\
        --model      swiss-ai/Apertus-8B-Instruct-2509 \\
        --vectors-dir output_apertus/contrast_vectors \\
        --layer      24 \\
        --sentences  sentences.json \\
        --output-dir output_token_viz/apertus_l24
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path("/users/sinievdben/scratch/personal/emotion_experiment")

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _patch_gemma4_tokenizer():
    """Patch transformers so Gemma 4's list-valued extra_special_tokens loads."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    _orig = PreTrainedTokenizerBase._set_model_specific_special_tokens
    def _patched(self, special_tokens):
        if isinstance(special_tokens, dict):
            _orig(self, special_tokens)
    PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched


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


def load_model(model_name: str, device: torch.device):
    _patch_gemma4_tokenizer()
    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def get_token_activations(
    text: str,
    tokenizer,
    model: nn.Module,
    layer: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    """
    Returns (tokens, activations) where:
      tokens      : list of decoded token strings, BOS removed
      activations : float32 array of shape (n_tokens, d_model),
                    each row L2-normalised so projections are cosine similarities
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    ids    = inputs["input_ids"][0]

    captured = {}
    def _hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["hidden"] = hs.detach().float()

    handle = _get_layer(model, layer).register_forward_hook(_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    hidden = captured["hidden"][0]                    # (seq_len, d_model)
    acts   = np.array(hidden.cpu().tolist(), dtype=np.float32)

    tokens = [tokenizer.decode([tok_id]) for tok_id in ids.tolist()]

    # Drop BOS token (index 0) 
    tokens = tokens[1:]
    acts   = acts[1:]

    # L2-normalise each token so projections become cosine similarities (±1).  removes residual-stream magnitude differences between models and
    # makes values directly comparable across models and layers.
    norms = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-10
    acts  = acts / norms

    return tokens, acts


def load_contrast_vectors(
    vectors_dir: Path,
    layer: int,
) -> tuple[list[str], np.ndarray]:
    """
    Returns (emotions, unit_vectors) where unit_vectors is (n_emotions, d_model),
    each row normalised to unit length.
    """
    mat_path = vectors_dir / f"layer_{layer}.npy"
    emo_path = vectors_dir / "emotions.json"
    if not mat_path.exists():
        raise FileNotFoundError(f"Contrast vector not found: {mat_path}")
    mat      = np.load(mat_path).astype(np.float32)          # (n_emotions, d_model)
    emotions = json.loads(emo_path.read_text())

    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    return emotions, mat / norms


def project_tokens(
    acts: np.ndarray,                    # (n_tokens, d_model)
    unit_vecs: np.ndarray,               # (n_emotions, d_model)
    emotion_names: list[str],
    target_emotions: list[str],
) -> dict[str, np.ndarray]:
    """
    Returns {emotion: per_token_projection_array} for each requested emotion.
    """
    result = {}
    for emo in target_emotions:
        key = emo.strip().lower().replace("_", " ")
        try:
            idx = next(i for i, e in enumerate(emotion_names)
                       if e.strip().lower().replace("_", " ") == key)
        except StopIteration:
            print(f"  WARNING: '{emo}' not found in contrast vectors, skipping")
            continue
        result[emo] = acts @ unit_vecs[idx]              # (n_tokens,)
    return result


def _clean_token(tok: str) -> str:
    """Make token strings printable in figures."""
    return tok.replace("▁", " ").replace("Ġ", " ").replace("\n", "↵").strip() or "·"


def fig_token_heatmap(
    tokens: list[str],
    projections: np.ndarray,           # (n_tokens,) for the primary emotion
    primary_emotion: str,
    sentence_label: str,
    model_name: str,
    layer: int,
    out_path: Path,
) -> None:
    """
    Draw each token as a coloured box; colour encodes projection strength.
    Diverging colormap: positive = red (emotion active), negative = blue.
    """
    labels   = [_clean_token(t) for t in tokens]
    n        = len(labels)
    abs_max  = max(abs(projections).max(), 1e-6)
    norm     = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap     = plt.get_cmap("RdBu_r")

    # Layout: wrap tokens into rows of ~12 each
    row_size = 12
    rows     = [labels[i:i+row_size] for i in range(0, n, row_size)]
    proj_rows= [projections[i:i+row_size] for i in range(0, n, row_size)]

    fig_w = row_size * 0.9 + 1
    fig_h = len(rows) * 0.8 + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, row_size)
    ax.set_ylim(-len(rows), 0)
    ax.axis("off")

    box_w, box_h = 0.88, 0.7
    for row_idx, (row_labels, row_proj) in enumerate(zip(rows, proj_rows)):
        for col_idx, (label, proj) in enumerate(zip(row_labels, row_proj)):
            color = cmap(norm(proj))
            rect  = mpatches.FancyBboxPatch(
                (col_idx + 0.05, -row_idx - box_h - 0.05),
                box_w, box_h,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="white", linewidth=0.5,
            )
            ax.add_patch(rect)
            # Choose text colour for contrast
            luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_color = "white" if luminance < 0.5 else "black"
            ax.text(col_idx + 0.05 + box_w/2, -row_idx - box_h/2 - 0.05,
                    label, ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="normal")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, orientation="vertical")
    cbar.set_label("cosine sim\n(mean-centred)", fontsize=8)

    ax.set_title(
        f'Token activations — "{primary_emotion}" vector\n'
        f'{model_name}  ·  layer {layer}  ·  {sentence_label}',
        fontsize=9, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_emotion_bar(
    projections_by_emotion: dict[str, np.ndarray],
    primary_emotion: str,
    sentence_label: str,
    sentence_text: str,
    model_name: str,
    layer: int,
    out_path: Path,
) -> None:
    """
    Bar chart: mean token projection for each emotion.
    Primary emotion highlighted; bars sorted descending.
    """
    means = {e: float(p.mean()) for e, p in projections_by_emotion.items()}
    # Sort: primary first, then descending by mean
    order = [primary_emotion] + sorted(
        [e for e in means if e != primary_emotion],
        key=lambda e: means.get(e, 0), reverse=True,
    )
    order = [e for e in order if e in means]

    values = [means[e] for e in order]
    colors = ["#E07B39" if e == primary_emotion else
              ("#6C8EBF" if means[e] > 0 else "#AAAAAA")
              for e in order]

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.7 + 1.5), 4))
    bars = ax.bar(range(len(order)), values, color=colors,
                  edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean token projection")
    ax.set_title(
        f'Emotion specificity — "{primary_emotion}" sentence\n'
        f'{model_name}  ·  layer {layer}',
        fontsize=9,
    )

    for bar, val in zip(bars, values):
        ypos = val + 0.001 if val >= 0 else val - 0.001
        va   = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{val:.3f}", ha="center", va=va, fontsize=7)

    short = sentence_text if len(sentence_text) < 80 else sentence_text[:77] + "..."
    fig.text(0.5, -0.02, f'"{short}"', ha="center", fontsize=7.5,
             style="italic", color="#555555")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_combined_heatmaps(
    all_results: list[dict],
    model_name: str,
    layer: int,
    out_path: Path,
) -> None:
    """
    Multi-panel figure: one token heatmap row per sentence, stacked vertically.
    Good for a single publication figure showing multiple examples.
    """
    n_sentences = len(all_results)
    row_size    = 14
    fig_w       = row_size * 0.78 + 1.5
    fig_h       = n_sentences * 2.2 + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdBu_r")

    for panel_idx, res in enumerate(all_results):
        tokens     = [_clean_token(t) for t in res["tokens"]]
        projections= res["primary_projections"]
        emotion    = res["primary_emotion"]
        label      = res["label"]
        n          = len(tokens)

        abs_max = max(abs(projections).max(), 1e-6)
        norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        ax = fig.add_axes([
            0.05,
            1 - (panel_idx + 1) / n_sentences + 0.01,
            0.88,
            0.85 / n_sentences,
        ])
        ax.set_xlim(0, min(n, row_size))
        ax.set_ylim(-1, 0)
        ax.axis("off")
        ax.set_title(f'"{emotion}"  ({label})', fontsize=8, loc="left", pad=3)

        # Only show first `row_size` tokens to keep layout uniform
        for col_idx, (tok, proj) in enumerate(zip(tokens[:row_size], projections[:row_size])):
            color     = cmap(norm(proj))
            luminance = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            rect = mpatches.FancyBboxPatch(
                (col_idx + 0.04, -0.85),
                0.88, 0.75,
                boxstyle="round,pad=0.03",
                facecolor=color, edgecolor="white", linewidth=0.4,
            )
            ax.add_patch(rect)
            ax.text(col_idx + 0.48, -0.475, tok,
                    ha="center", va="center", fontsize=6.5,
                    color="white" if luminance < 0.5 else "black")

        if n > row_size:
            ax.text(row_size - 0.1, -0.475, f"… +{n-row_size}",
                    ha="right", va="center", fontsize=6, color="#888888")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=mcolors.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.1, 0.015, 0.8])
    fig.colorbar(sm, cax=cbar_ax, label="cosine sim")

    fig.suptitle(
        f"Token-level emotion vector activations\n{model_name}  ·  layer {layer}",
        fontsize=10, y=1.01,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        type=str,  required=True)
    parser.add_argument("--vectors-dir",  type=Path, required=True,
                        help="Directory containing layer_N.npy contrast vectors "
                             "and emotions.json (output of analyze_emotion_vectors.py)")
    parser.add_argument("--layer",        type=int,  required=True)
    parser.add_argument("--sentences",    type=Path,
                        default=SCRIPT_DIR / "sentences.json")
    parser.add_argument("--output-dir",   type=Path, required=True)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load sentences, model, contrast vectors
    sentences = json.loads(args.sentences.read_text())
    print(f"Loaded {len(sentences)} sentences from {args.sentences.name}")

    print(f"\nLoading contrast vectors from {args.vectors_dir} (layer {args.layer}) ...")
    emotion_names, unit_vecs = load_contrast_vectors(args.vectors_dir, args.layer)
    print(f"  {len(emotion_names)} emotion vectors  (d={unit_vecs.shape[1]})")

    print()
    device   = torch.device(args.device)
    tokenizer, model = load_model(args.model, device)

    all_results = []

    for entry in sentences:
        label    = entry["label"]
        text     = entry["text"]
        primary  = entry["primary_emotion"]
        compare  = entry["comparison_emotions"]
        all_emos = [primary] + [e for e in compare if e != primary]

        print(f"\n[{label}]  primary={primary}")
        print(f"  text: {text[:80]}{'...' if len(text) > 80 else ''}")

        tokens, acts = get_token_activations(
            text, tokenizer, model, args.layer, device)
        print(f"  tokens: {len(tokens)}")

        proj_by_emotion = project_tokens(acts, unit_vecs, emotion_names, all_emos)

        if primary not in proj_by_emotion:
            print(f"  WARNING: primary emotion '{primary}' not found, skipping")
            continue

        primary_proj = proj_by_emotion[primary]

        # Mean-centre all projections per sentence so the sign is always
        # interpretable: positive (red) = more emotion than sentence average,
        # negative (blue) = less. This removes the arbitrary sign offset of
        # the contrast vector and makes values comparable across models.
        sentence_mean = np.mean(np.stack(list(proj_by_emotion.values())))
        proj_by_emotion = {e: p - sentence_mean for e, p in proj_by_emotion.items()}
        primary_proj = proj_by_emotion[primary]

        # Per-sentence heatmap
        fig_token_heatmap(
            tokens, primary_proj, primary, label, args.model, args.layer,
            out_path=args.output_dir / f"{label}_token_heatmap.pdf",
        )

        # Per-sentence emotion bar chart
        fig_emotion_bar(
            proj_by_emotion, primary, label, text, args.model, args.layer,
            out_path=args.output_dir / f"{label}_emotion_bar.pdf",
        )

        all_results.append({
            "label": label,
            "text": text,
            "primary_emotion": primary,
            "tokens": tokens,
            "primary_projections": primary_proj,
            "mean_projections": {e: float(p.mean()) for e, p in proj_by_emotion.items()},
            "max_projections":  {e: float(p.max())  for e, p in proj_by_emotion.items()},
        })

    if all_results:
        print("\nGenerating combined heatmap figure ...")
        fig_combined_heatmaps(
            all_results, args.model, args.layer,
            out_path=args.output_dir / "fig_combined_token_heatmaps.pdf",
        )

    summary = [
        {k: v.tolist() if isinstance(v, np.ndarray) else v
         for k, v in r.items()}
        for r in all_results
    ]
    out_json = args.output_dir / "token_projections.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nProjections saved → {out_json}")
    print(f"All figures → {args.output_dir}")


if __name__ == "__main__":
    main()
