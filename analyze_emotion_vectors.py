#!/usr/bin/env python3
"""
Emotion vector analysis following the Anthropic emotion-vectors methodology,
applied to Apertus 8B + BatchTopK SAEs.

Produces figures:
  fig1_cosine_similarity.pdf   — pairwise cosine sim heatmap (contrast vectors)
  fig2_pca.pdf                 — PCA scatter + valence/arousal correlation panels
  fig3_umap.pdf                — UMAP coloured by k-means cluster
  fig4_cross_layer.pdf         — CKA matrix + valence-direction stability

Usage (inside container):
    python3 analyze_emotion_vectors.py [--vectors-dir ...] [--output-dir ...]
"""

import argparse
import csv
import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path("/users/sinievdben/scratch/personal/emotion_experiment")
VECTORS_DIR = SCRIPT_DIR / "output/emotion_vectors"
OUTPUT_DIR  = SCRIPT_DIR / "output/analysis"
VAD_CSV     = SCRIPT_DIR / "emotion_valence_arousal_nrc.csv"

LAYERS = [12, 16, 17, 18, 19, 20]


# ─── NRC VAD ratings ──────────────────────────────────────────────────────────

def load_circumplex(csv_path: Path) -> dict[str, tuple[float, float]]:
    """
    Load valence/arousal ratings from the NRC VAD CSV.

    Expected columns: emotion, valence, arousal  (plus optional others).
    Emotion names in the CSV are matched case-insensitively; spaces and
    underscores are treated as equivalent so that 'on edge' matches 'on_edge'.
    """
    def normalise(s: str) -> str:
        return s.strip().lower().replace("_", " ")

    circumplex: dict[str, tuple[float, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalise(row["emotion"])
            circumplex[key] = (float(row["valence"]), float(row["arousal"]))

    print(f"Loaded {len(circumplex)} VAD entries from {csv_path.name}")
    return circumplex


def lookup_vad(
    emotion: str,
    circumplex: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Return (valence, arousal) for an emotion, or None if not found."""
    key = emotion.strip().lower().replace("_", " ")
    return circumplex.get(key)

# ─── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

VALENCE_CMAP = plt.get_cmap("RdYlGn")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_vectors(
    vectors_dir: Path,
    layers: list[int],
    circumplex: dict[str, tuple[float, float]],
    use_resid: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Returns vecs[layer][emotion] = float32 array.

    use_resid=False (default): SAE feature vectors (d_sae=65536).
      Priority: layer_N_projected.npy > layer_N.npy

    use_resid=True: residual-stream vectors (d_model=4096).
      Priority: layer_N_resid_projected.npy > layer_N_resid.npy

    Only includes emotions present in ALL requested layers AND with VAD ratings.
    """
    vecs: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}

    def _has_rating(name: str) -> bool:
        return lookup_vad(name, circumplex) is not None

    emotion_dirs = sorted([
        d for d in vectors_dir.iterdir()
        if d.is_dir() and _has_rating(d.name)
    ])

    for emo_dir in emotion_dirs:
        emotion = emo_dir.name
        for layer in layers:
            if use_resid:
                projected = emo_dir / f"layer_{layer}_resid_projected.npy"
                fallback  = emo_dir / f"layer_{layer}_resid.npy"
            else:
                projected = emo_dir / f"layer_{layer}_projected.npy"
                fallback  = emo_dir / f"layer_{layer}.npy"
            path = projected if projected.exists() else fallback
            if path.exists():
                vecs[layer][emotion] = np.load(path).astype(np.float32)

    # Determine which variant was actually loaded for the status message
    first_dir = next((d for d in vectors_dir.iterdir() if d.is_dir()), None)
    if use_resid:
        space = "residual stream"
        variant = "projected" if (
            first_dir and (first_dir / f"layer_{layers[0]}_resid_projected.npy").exists()
        ) else "raw"
    else:
        space = "SAE"
        variant = "projected" if (
            first_dir and (first_dir / f"layer_{layers[0]}_projected.npy").exists()
        ) else "raw"
    print(f"Using {variant} {space} emotion vectors")

    # Keep only emotions present in every layer
    common = set(vecs[layers[0]].keys())
    for l in layers[1:]:
        common &= set(vecs[l].keys())

    for l in layers:
        vecs[l] = {e: vecs[l][e] for e in sorted(common)}

    dim = next(iter(vecs[layers[0]].values())).shape[0]
    print(f"Loaded {len(common)} emotions × {len(layers)} layers  (dim={dim})")
    return vecs


def build_matrices(raw: dict[int, dict[str, np.ndarray]]) -> dict[int, np.ndarray]:
    """
    Returns mat[layer] = float32 array of shape (n_emotions, d_sae),
    row-order == sorted(emotions).
    """
    return {
        l: np.stack(list(vecs.values()), axis=0)
        for l, vecs in raw.items()
    }


def contrast_vectors(mat: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Subtract per-layer cross-emotion mean → emotion-specific directions."""
    return {l: M - M.mean(axis=0, keepdims=True) for l, M in mat.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Cosine similarity heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_cosine_heatmap(
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
    layer: int,
    out_path: Path,
) -> None:
    C = normalize(contrast[layer])            # unit-norm rows
    sim = C @ C.T                             # (n, n) cosine similarities

    # Sort emotions by valence so the heatmap shows a natural gradient
    valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])
    order = np.argsort(valences)
    sim_sorted = sim[np.ix_(order, order)]
    labels_sorted = [emotions[i] for i in order]

    n = len(emotions)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_sorted, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, label="cosine similarity")

    tick_step = max(1, n // 20)
    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels_sorted[i] for i in ticks], rotation=90)
    ax.set_yticklabels([labels_sorted[i] for i in ticks])

    ax.set_title(f"Pairwise cosine similarity of contrast vectors (layer {layer})\n"
                 "sorted by valence: negative → positive")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — PCA + valence/arousal correlation
# ═══════════════════════════════════════════════════════════════════════════════

def fig_pca(
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
    layer: int,
    out_path: Path,
) -> dict:
    valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])
    arousals = np.array([lookup_vad(e, circumplex)[1] for e in emotions])

    pca = PCA(n_components=min(10, len(emotions) - 1))
    coords = pca.fit_transform(contrast[layer])   # (n_emotions, n_components)

    pc1, pc2 = coords[:, 0], coords[:, 1]

    # Flip sign so PC1 ~ valence (positive = positive valence)
    if pearsonr(pc1, valences)[0] < 0:
        pc1 = -pc1
    if pearsonr(pc2, arousals)[0] < 0:
        pc2 = -pc2

    r_val, p_val = pearsonr(pc1, valences)
    r_aro, p_aro = pearsonr(pc2, arousals)
    rho_val = spearmanr(pc1, valences).correlation
    rho_aro = spearmanr(pc2, arousals).correlation

    ev = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(14, 4.5), constrained_layout=True)
    gs  = fig.add_gridspec(1, 3)

    # Panel A: scatter PC1 vs PC2
    ax0 = fig.add_subplot(gs[0])
    sc  = ax0.scatter(pc1, pc2,
                      c=valences, cmap=VALENCE_CMAP, vmin=-1, vmax=1,
                      s=40, edgecolors="k", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax0, fraction=0.05, label="valence")
    # Annotate a few anchor emotions
    anchors = {"ecstatic", "depressed", "furious", "serene", "calm", "panicked",
               "joyful", "gloomy", "excited", "peaceful"}
    for i, e in enumerate(emotions):
        if e in anchors:
            ax0.annotate(e, (pc1[i], pc2[i]), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    ax0.axhline(0, color="grey", lw=0.5, ls="--")
    ax0.axvline(0, color="grey", lw=0.5, ls="--")
    ax0.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax0.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax0.set_title(f"Emotion geometry in PC space\n(layer {layer})")

    # Panel B: PC1 vs valence
    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(valences, pc1, c=valences, cmap=VALENCE_CMAP,
                vmin=-1, vmax=1, s=35, edgecolors="k", linewidths=0.3)
    m, b = np.polyfit(valences, pc1, 1)
    xline = np.linspace(valences.min(), valences.max(), 100)
    ax1.plot(xline, m*xline + b, "k--", lw=1)
    ax1.set_xlabel("valence (Russell circumplex)")
    ax1.set_ylabel("PC1 projection")
    ax1.set_title(f"PC1 ↔ valence\nr={r_val:.3f}, ρ={rho_val:.3f} (p={p_val:.2e})")

    # Panel C: PC2 vs arousal
    ax2 = fig.add_subplot(gs[2])
    ax2.scatter(arousals, pc2, c=arousals, cmap="plasma",
                vmin=-1, vmax=1, s=35, edgecolors="k", linewidths=0.3)
    m, b = np.polyfit(arousals, pc2, 1)
    xline = np.linspace(arousals.min(), arousals.max(), 100)
    ax2.plot(xline, m*xline + b, "k--", lw=1)
    ax2.set_xlabel("arousal (NRC VAD)")
    ax2.set_ylabel("PC2 projection")
    ax2.set_title(f"PC2 ↔ arousal\nr={r_aro:.3f}, ρ={rho_aro:.3f} (p={p_aro:.2e})")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")

    return {
        "layer": layer,
        "pc1_valence_r": r_val, "pc1_valence_p": p_val,
        "pc1_valence_rho": rho_val,
        "pc2_arousal_r": r_aro, "pc2_arousal_p": p_aro,
        "pc2_arousal_rho": rho_aro,
        "explained_variance_ratio": ev.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — UMAP + k-means clustering
# ═══════════════════════════════════════════════════════════════════════════════

def fig_umap(
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
    layer: int,
    k: int,
    out_path: Path,
) -> None:
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, n_neighbors=10, min_dist=0.2,
                       metric="cosine", random_state=42)
        dim_label = "UMAP"
    except ImportError:
        print("  umap-learn not available — falling back to t-SNE")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, metric="cosine", random_state=42,
                       perplexity=min(30, len(emotions) - 1), init="pca")
        dim_label = "t-SNE"

    valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(contrast[layer])

    coords = reducer.fit_transform(contrast[layer])

    cluster_colors = plt.get_cmap("tab10")(np.linspace(0, 1, k))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (color_by, title, cmap_arg) in zip(axes, [
        ("cluster", f"{dim_label} — k-means clusters (k={k}, layer {layer})", None),
        ("valence", f"{dim_label} — coloured by valence (layer {layer})", "valence"),
    ]):
        if cmap_arg == "valence":
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=valences, cmap=VALENCE_CMAP, vmin=-1, vmax=1,
                            s=45, edgecolors="k", linewidths=0.3)
            plt.colorbar(sc, ax=ax, fraction=0.05, label="valence")
        else:
            for cl in range(k):
                mask = labels == cl
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           color=cluster_colors[cl], s=45,
                           edgecolors="k", linewidths=0.3,
                           label=f"C{cl}")
            ax.legend(markerscale=0.8, ncol=2, fontsize=7)

        for i, e in enumerate(emotions):
            ax.annotate(e, coords[i], fontsize=5,
                        xytext=(2, 2), textcoords="offset points", alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel(f"{dim_label} 1")
        ax.set_ylabel(f"{dim_label} 2")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")

    # Print cluster contents
    print(f"\n  K-means clusters (k={k}, layer {layer}):")
    for cl in range(k):
        members = [emotions[i] for i in range(len(emotions)) if labels[i] == cl]
        print(f"    C{cl}: {', '.join(members)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cross-layer analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA between two (n_samples, d) matrices.
    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    Centering is applied to remove mean.
    """
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n      # centering matrix

    Xc = H @ X
    Yc = H @ Y

    dot = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    norm = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")

    return float(dot / (norm + 1e-10))


def _valence_direction(
    contrast_mat: np.ndarray,
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Unit vector in the valence direction: linear regression of valence → contrast."""
    valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])
    # Least-squares: find w such that contrast @ w ≈ valences
    w, _, _, _ = np.linalg.lstsq(contrast_mat, valences, rcond=None)
    return w / (np.linalg.norm(w) + 1e-10)


def fig_cross_layer(
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
    layers: list[int],
    out_path: Path,
) -> dict:
    n = len(layers)

    # ── CKA matrix ────────────────────────────────────────────────────────────
    cka_mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            cka_mat[i, j] = _linear_cka(contrast[li], contrast[lj])

    # ── Valence-direction cosine similarity across layers ─────────────────────
    val_dirs = {l: _valence_direction(contrast[l], emotions, circumplex) for l in layers}

    # Cosine sim of valence direction between adjacent layers
    adj_sim = []
    for i in range(len(layers) - 1):
        a, b = val_dirs[layers[i]], val_dirs[layers[i + 1]]
        adj_sim.append(float(np.dot(a, b)))

    # All-pairs valence direction similarities
    val_sim_mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            val_sim_mat[i, j] = float(np.dot(val_dirs[li], val_dirs[lj]))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    layer_labels = [str(l) for l in layers]

    # Panel A: CKA matrix
    ax = axes[0]
    im = ax.imshow(cka_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.05, label="linear CKA")
    ax.set_xticks(range(n)); ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(layer_labels)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title("Representational similarity\n(linear CKA, contrast vectors)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cka_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cka_mat[i,j] < 0.5 else "black")

    # Panel B: Valence-direction cosine similarity matrix
    ax = axes[1]
    im = ax.imshow(val_sim_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.05, label="cosine sim")
    ax.set_xticks(range(n)); ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(layer_labels)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title("Valence direction alignment\n(cosine similarity across layers)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{val_sim_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7)

    # Panel C: Adjacent-layer valence-direction similarity (line plot)
    ax = axes[2]
    mid_layers = [(layers[i] + layers[i+1]) / 2 for i in range(len(layers) - 1)]
    ax.plot(mid_layers, adj_sim, "o-", color="#2196F3", lw=2, ms=6)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xticks(layers)
    ax.set_xlabel("layer (midpoint between pair)")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Valence direction: adjacent-layer stability")
    ax.set_ylim(-1, 1)
    for x, y in zip(mid_layers, adj_sim):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)

    fig.suptitle("Cross-layer emotion geometry analysis (Apertus 8B)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")

    return {
        "cka_matrix": cka_mat.tolist(),
        "valence_direction_sim_matrix": val_sim_mat.tolist(),
        "adjacent_layer_valence_sim": {
            f"{layers[i]}-{layers[i+1]}": adj_sim[i]
            for i in range(len(layers) - 1)
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(
    raw: dict[int, dict[str, np.ndarray]],
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    layers: list[int],
) -> dict:
    summary = {"layers": {}}
    for l in layers:
        C = normalize(contrast[l])
        sim = C @ C.T
        np.fill_diagonal(sim, np.nan)

        # Find the most opposite pairs (lowest cosine sim)
        idx = np.unravel_index(np.nanargmin(sim), sim.shape)
        most_opposite = (emotions[idx[0]], emotions[idx[1]], float(sim[idx]))

        # Sparsity of raw vectors
        sparsity = float((raw[l][emotions[0]] == 0).mean())  # use first as proxy

        summary["layers"][str(l)] = {
            "mean_cosine_sim": float(np.nanmean(sim)),
            "std_cosine_sim":  float(np.nanstd(sim)),
            "most_opposite_pair": most_opposite,
            "raw_vector_sparsity": sparsity,
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors-dir", type=Path, default=VECTORS_DIR)
    parser.add_argument("--output-dir",  type=Path, default=OUTPUT_DIR)
    parser.add_argument("--vad-csv",     type=Path, default=VAD_CSV)
    parser.add_argument("--layers",      type=int, nargs="+", default=LAYERS)
    parser.add_argument("--heatmap-layer", type=int, default=18,
                        help="Which layer to use for the cosine heatmap and UMAP (default: 18)")
    parser.add_argument("--kmeans-k",    type=int, default=10)
    parser.add_argument("--use-resid",   action="store_true",
                        help="Analyse residual-stream vectors instead of SAE feature vectors")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load VAD ratings ──────────────────────────────────────────────────────
    circumplex = load_circumplex(args.vad_csv)

    # ── Load vectors ──────────────────────────────────────────────────────────
    print("Loading vectors ...")
    raw = load_vectors(args.vectors_dir, args.layers, circumplex, use_resid=args.use_resid)
    emotions = sorted(next(iter(raw.values())).keys())
    mat = build_matrices(raw)
    contrast = contrast_vectors(mat)

    # Save contrast vectors as npy for downstream use
    contrast_dir = args.output_dir / "contrast_vectors"
    contrast_dir.mkdir(exist_ok=True)
    for l, C in contrast.items():
        np.save(contrast_dir / f"layer_{l}.npy", C)
    with open(contrast_dir / "emotions.json", "w") as f:
        json.dump(emotions, f)
    print(f"Saved contrast vectors → {contrast_dir}")

    # ── Figure 1: Cosine heatmap ───────────────────────────────────────────────
    print("\nFigure 1: cosine similarity heatmap ...")
    for l in args.layers:
        fig_cosine_heatmap(
            contrast, emotions, circumplex,
            layer=l,
            out_path=args.output_dir / f"fig1_cosine_similarity_layer{l}.pdf",
        )
        print(f"  saved fig1_cosine_similarity_layer{l}.pdf")

    # ── Figure 2: PCA ─────────────────────────────────────────────────────────
    print("Figure 2: PCA + valence/arousal correlation ...")
    pca_results = {}
    for l in args.layers:
        r = fig_pca(
            contrast, emotions, circumplex,
            layer=l,
            out_path=args.output_dir / f"fig2_pca_layer{l}.pdf",
        )
        pca_results[str(l)] = r
        print(f"  layer {l}: PC1↔valence r={r['pc1_valence_r']:.3f}  "
              f"PC2↔arousal r={r['pc2_arousal_r']:.3f}")

    # ── Figure 3: UMAP ────────────────────────────────────────────────────────
    print("Figure 3: UMAP + k-means ...")
    fig_umap(
        contrast, emotions, circumplex,
        layer=args.heatmap_layer,
        k=args.kmeans_k,
        out_path=args.output_dir / "fig3_umap.pdf",
    )

    # ── Figure 4: Cross-layer ─────────────────────────────────────────────────
    print("Figure 4: cross-layer CKA + valence stability ...")
    cross_layer_results = fig_cross_layer(
        contrast, emotions, circumplex, args.layers,
        out_path=args.output_dir / "fig4_cross_layer.pdf",
    )

    # ── Summary stats ─────────────────────────────────────────────────────────
    summary = compute_summary(raw, contrast, emotions, args.layers)
    summary["pca"] = pca_results
    summary["cross_layer"] = cross_layer_results
    summary["n_emotions"] = len(emotions)
    summary["emotions"] = emotions

    summary_path = args.output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Emotions: {len(emotions)}   Layers: {args.layers}")
    print()
    print(f"  {'Layer':<8} {'mean cosine':>12} {'PC1↔val r':>12} {'PC2↔aro r':>12}")
    print(f"  {'-'*44}")
    for l in args.layers:
        s = summary["layers"][str(l)]
        p = pca_results.get(str(l), {})
        print(f"  {l:<8} {s['mean_cosine_sim']:>12.3f} "
              f"{p.get('pc1_valence_r', float('nan')):>12.3f} "
              f"{p.get('pc2_arousal_r', float('nan')):>12.3f}")
    print()
    print(f"  Adjacent-layer valence stability:")
    for pair, sim in cross_layer_results["adjacent_layer_valence_sim"].items():
        print(f"    layers {pair}: {sim:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
