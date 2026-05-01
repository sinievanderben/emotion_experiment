#!/usr/bin/env python3
"""
Cross-model geometry analysis: Gemma 4 8B vs Apertus 8B.

Figures produced in OUTPUT_DIR:
  fig_valence_trajectory.pdf   — PC1↔valence r across layers, both models
  fig_arousal_trajectory.pdf   — PC2↔arousal r across layers, both models
  fig_cka_gemma_mid.pdf        — CKA heatmap Gemma L16-24
  fig_cka_gemma_late.pdf       — CKA heatmap Gemma L28-40
  fig_cka_apertus_mid.pdf      — CKA heatmap Apertus available mid layers
  fig_cka_apertus_late.pdf     — CKA heatmap Apertus available late layers
  fig_pca_gemma_l16.pdf        — PCA scatter Gemma L16, coloured by valence
  fig_pca_apertus_l26.pdf      — PCA scatter Apertus L26, coloured by valence

Usage:
    python3 analyze_cross_model_geometry.py [--gemma-dir ...] [--apertus-dir ...] [--output-dir ...]
"""

import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path("/users/sinievdben/scratch/personal/emotion_experiment")
VAD_CSV      = SCRIPT_DIR / "emotion_valence_arousal_nrc.csv"

GEMMA_DIR    = SCRIPT_DIR / "output_gemma/emotion_vectors"
APERTUS_DIR  = SCRIPT_DIR / "output_apertus/emotion_vectors"

GEMMA_LAYERS_REQUESTED   = [16, 18, 20, 24, 28, 32, 36, 40]
APERTUS_LAYERS_REQUESTED = [12, 16, 18, 20, 22, 24, 26, 28, 30]

GEMMA_MID_LAYERS   = [16, 18, 20, 24]
GEMMA_LATE_LAYERS  = [28, 32, 36, 40]
APERTUS_MID_LAYERS = [12, 16, 18, 20]       # 22, 24 will be skipped if absent
APERTUS_LATE_LAYERS = [22, 24, 26, 28, 30]  # same

GEMMA_BEST_LAYER   = 16
APERTUS_BEST_LAYER = 26

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
GEMMA_COLOR   = "#E07B39"
APERTUS_COLOR = "#3976C8"


def load_circumplex(csv_path: Path) -> dict[str, tuple[float, float]]:
    def normalise(s: str) -> str:
        return s.strip().lower().replace("_", " ")

    circumplex: dict[str, tuple[float, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalise(row["emotion"])
            circumplex[key] = (float(row["valence"]), float(row["arousal"]))
    print(f"Loaded {len(circumplex)} VAD entries")
    return circumplex


def lookup_vad(
    emotion: str,
    circumplex: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    key = emotion.strip().lower().replace("_", " ")
    return circumplex.get(key)


def _layer_file(emo_dir: Path, layer: int) -> Path | None:
    """Return the best available resid vector file for this layer, or None."""
    projected = emo_dir / f"layer_{layer}_resid_projected.npy"
    fallback   = emo_dir / f"layer_{layer}_resid.npy"
    if projected.exists():
        return projected
    if fallback.exists():
        return fallback
    return None


def load_vectors(
    vectors_dir: Path,
    requested_layers: list[int],
    circumplex: dict[str, tuple[float, float]],
) -> tuple[dict[int, dict[str, np.ndarray]], list[int]]:
    """
    Load resid vectors for all requested layers that actually exist on disk.
    Returns (vecs, available_layers) where available_layers ⊆ requested_layers.
    Only emotions present in ALL available layers AND with VAD ratings are kept.
    """
    emotion_dirs = sorted([
        d for d in vectors_dir.iterdir()
        if d.is_dir() and lookup_vad(d.name, circumplex) is not None
    ])

    # Determine which requested layers exist (check first emotion dir)
    available_layers = []
    if emotion_dirs:
        for layer in requested_layers:
            if _layer_file(emotion_dirs[0], layer) is not None:
                available_layers.append(layer)
            else:
                print(f"  WARNING: layer {layer} not found in {vectors_dir.name}, skipping")

    vecs: dict[int, dict[str, np.ndarray]] = {l: {} for l in available_layers}
    for emo_dir in emotion_dirs:
        emotion = emo_dir.name
        for layer in available_layers:
            path = _layer_file(emo_dir, layer)
            if path is not None:
                vecs[layer][emotion] = np.load(path).astype(np.float32)

    # Keep only emotions present in every available layer
    if available_layers:
        common = set(vecs[available_layers[0]].keys())
        for l in available_layers[1:]:
            common &= set(vecs[l].keys())
        for l in available_layers:
            vecs[l] = {e: vecs[l][e] for e in sorted(common)}
        dim = next(iter(vecs[available_layers[0]].values())).shape[0]
        print(f"  {len(common)} emotions × {len(available_layers)} layers (dim={dim})")
    else:
        print("  WARNING: no layers found")

    return vecs, available_layers


def build_contrast(vecs: dict[int, dict[str, np.ndarray]]) -> dict[int, np.ndarray]:
    """Stack per-layer emotion vectors and subtract cross-emotion mean."""
    contrast = {}
    for l, ev in vecs.items():
        M = np.stack(list(ev.values()), axis=0)
        contrast[l] = M - M.mean(axis=0, keepdims=True)
    return contrast


def run_pca_layer(
    contrast_mat: np.ndarray,
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
) -> dict:
    valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])
    arousals = np.array([lookup_vad(e, circumplex)[1] for e in emotions])

    pca = PCA(n_components=min(10, len(emotions) - 1))
    coords = pca.fit_transform(contrast_mat)

    pc1, pc2 = coords[:, 0], coords[:, 1]

    # Canonicalise sign so PC1 ~ valence, PC2 ~ arousal
    if pearsonr(pc1, valences)[0] < 0:
        pc1 = -pc1
    if pearsonr(pc2, arousals)[0] < 0:
        pc2 = -pc2

    r_val, p_val   = pearsonr(pc1, valences)
    r_aro, p_aro   = pearsonr(pc2, arousals)
    rho_val        = spearmanr(pc1, valences).correlation
    rho_aro        = spearmanr(pc2, arousals).correlation
    ev             = pca.explained_variance_ratio_

    return {
        "pc1": pc1, "pc2": pc2,
        "valences": valences, "arousals": arousals,
        "r_val": r_val, "p_val": p_val, "rho_val": rho_val,
        "r_aro": r_aro, "p_aro": p_aro, "rho_aro": rho_aro,
        "ev": ev,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Xc, Yc = H @ X, H @ Y
    dot  = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    norm = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(dot / (norm + 1e-10))


def compute_cka_matrix(contrast: dict[int, np.ndarray], layers: list[int]) -> np.ndarray:
    n = len(layers)
    mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            mat[i, j] = linear_cka(contrast[li], contrast[lj])
    return mat


def fig_trajectory(
    gemma_layers: list[int],
    gemma_pca: dict[int, dict],
    apertus_layers: list[int],
    apertus_pca: dict[int, dict],
    metric: str,                 # "r_val" | "r_aro"
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))

    gx = gemma_layers
    gy = [gemma_pca[l][metric] for l in gx]
    ax.plot(gx, gy, "o-", color=GEMMA_COLOR, lw=2, ms=6, label="Gemma 4 8B")

    ax2 = apertus_layers
    ay  = [apertus_pca[l][metric] for l in ax2]
    ax.plot(ax2, ay, "s--", color=APERTUS_COLOR, lw=2, ms=6, label="Apertus 8B")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_cka_heatmap(
    cka_mat: np.ndarray,
    layers: list[int],
    model_name: str,
    out_path: Path,
) -> None:
    n = len(layers)
    fig, ax = plt.subplots(figsize=(0.7 * n + 1.5, 0.7 * n + 1.2))
    im = ax.imshow(cka_mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.05, label="linear CKA")
    labels = [str(l) for l in layers]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title(f"{model_name} — representational similarity (CKA)\nlayers {layers[0]}–{layers[-1]}")
    for i in range(n):
        for j in range(n):
            v = cka_mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v < 0.5 else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_pca_scatter(
    pca_result: dict,
    emotions: list[str],
    layer: int,
    model_name: str,
    out_path: Path,
) -> None:
    pc1     = pca_result["pc1"]
    pc2     = pca_result["pc2"]
    valences = pca_result["valences"]
    ev      = pca_result["ev"]
    r_val   = pca_result["r_val"]
    r_aro   = pca_result["r_aro"]
    rho_val = pca_result["rho_val"]
    rho_aro = pca_result["rho_aro"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    # Panel A: PC1 vs PC2 scatter
    ax = axes[0]
    sc = ax.scatter(pc1, pc2, c=valences, cmap=VALENCE_CMAP, vmin=-1, vmax=1,
                    s=45, edgecolors="k", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.05, label="valence")
    anchors = {"ecstatic", "depressed", "furious", "serene", "calm", "panicked",
               "joyful", "gloomy", "excited", "peaceful", "afraid", "happy",
               "angry", "sad", "surprised", "disgusted"}
    for i, e in enumerate(emotions):
        if e in anchors:
            ax.annotate(e, (pc1[i], pc2[i]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title(f"{model_name} L{layer} — emotion geometry")

    # Panel B: PC1 vs valence
    ax = axes[1]
    arousals = pca_result["arousals"]
    ax.scatter(valences, pc1, c=valences, cmap=VALENCE_CMAP, vmin=-1, vmax=1,
               s=35, edgecolors="k", linewidths=0.3)
    m, b = np.polyfit(valences, pc1, 1)
    xline = np.linspace(valences.min(), valences.max(), 100)
    ax.plot(xline, m*xline + b, "k--", lw=1)
    ax.set_xlabel("valence (NRC VAD)")
    ax.set_ylabel("PC1 projection")
    ax.set_title(f"PC1 ↔ valence\nr={r_val:.3f}, ρ={rho_val:.3f}")

    # Panel C: PC2 vs arousal
    ax = axes[2]
    ax.scatter(arousals, pc2, c=arousals, cmap="plasma", vmin=-1, vmax=1,
               s=35, edgecolors="k", linewidths=0.3)
    m, b = np.polyfit(arousals, pc2, 1)
    xline = np.linspace(arousals.min(), arousals.max(), 100)
    ax.plot(xline, m*xline + b, "k--", lw=1)
    ax.set_xlabel("arousal (NRC VAD)")
    ax.set_ylabel("PC2 projection")
    ax.set_title(f"PC2 ↔ arousal\nr={r_aro:.3f}, ρ={rho_aro:.3f}")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def fig_valence_direction_sim(
    contrast: dict[int, np.ndarray],
    emotions: list[str],
    circumplex: dict[str, tuple[float, float]],
    layers: list[int],
    model_name: str,
    out_path: Path,
) -> None:
    """Cosine similarity matrix of the valence direction vector across layers."""
    def valence_direction(mat: np.ndarray) -> np.ndarray:
        valences = np.array([lookup_vad(e, circumplex)[0] for e in emotions])
        w, _, _, _ = np.linalg.lstsq(mat, valences, rcond=None)
        return w / (np.linalg.norm(w) + 1e-10)

    val_dirs = {l: valence_direction(contrast[l]) for l in layers}
    n = len(layers)
    sim_mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            sim_mat[i, j] = float(np.dot(val_dirs[li], val_dirs[lj]))

    fig, ax = plt.subplots(figsize=(0.7 * n + 1.5, 0.7 * n + 1.2))
    im = ax.imshow(sim_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.05, label="cosine sim")
    labels = [str(l) for l in layers]
    ax.set_xticks(range(n)); ax.set_xticklabels(labels)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    ax.set_xlabel("layer"); ax.set_ylabel("layer")
    ax.set_title(f"{model_name} — valence direction alignment\n(cosine sim across layers)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-dir",   type=Path, default=GEMMA_DIR)
    parser.add_argument("--apertus-dir", type=Path, default=APERTUS_DIR)
    parser.add_argument("--output-dir",  type=Path, default=SCRIPT_DIR / "output_cross_model")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    circumplex = load_circumplex(VAD_CSV)

    print("\nLoading Gemma 4 8B vectors ...")
    gemma_vecs, gemma_layers = load_vectors(args.gemma_dir, GEMMA_LAYERS_REQUESTED, circumplex)
    gemma_emotions = sorted(next(iter(gemma_vecs.values())).keys())
    gemma_contrast = build_contrast(gemma_vecs)

    print("\nLoading Apertus 8B vectors ...")
    apertus_vecs, apertus_layers = load_vectors(args.apertus_dir, APERTUS_LAYERS_REQUESTED, circumplex)
    apertus_emotions = sorted(next(iter(apertus_vecs.values())).keys())
    apertus_contrast = build_contrast(apertus_vecs)

    print("\nRunning PCA ...")
    gemma_pca   = {l: run_pca_layer(gemma_contrast[l],   gemma_emotions,   circumplex) for l in gemma_layers}
    apertus_pca = {l: run_pca_layer(apertus_contrast[l], apertus_emotions, circumplex) for l in apertus_layers}

    for l in gemma_layers:
        p = gemma_pca[l]
        print(f"  Gemma   L{l:2d}: PC1↔val r={p['r_val']:.3f}  PC2↔aro r={p['r_aro']:.3f}")
    for l in apertus_layers:
        p = apertus_pca[l]
        print(f"  Apertus L{l:2d}: PC1↔val r={p['r_val']:.3f}  PC2↔aro r={p['r_aro']:.3f}")

    print("\nGenerating trajectory figures ...")
    fig_trajectory(
        gemma_layers, gemma_pca, apertus_layers, apertus_pca,
        metric="r_val",
        ylabel="Pearson r  (PC1 ↔ valence)",
        title="Valence encoding trajectory across layers",
        out_path=args.output_dir / "fig_valence_trajectory.pdf",
    )
    fig_trajectory(
        gemma_layers, gemma_pca, apertus_layers, apertus_pca,
        metric="r_aro",
        ylabel="Pearson r  (PC2 ↔ arousal)",
        title="Arousal encoding trajectory across layers",
        out_path=args.output_dir / "fig_arousal_trajectory.pdf",
    )

    print("\nComputing CKA matrices ...")

    def available(requested: list[int], actual: list[int]) -> list[int]:
        return [l for l in requested if l in actual]

    gemma_mid   = available(GEMMA_MID_LAYERS,    gemma_layers)
    gemma_late  = available(GEMMA_LATE_LAYERS,   gemma_layers)
    ap_mid      = available(APERTUS_MID_LAYERS,  apertus_layers)
    ap_late     = available(APERTUS_LATE_LAYERS, apertus_layers)

    for layers_subset, contrast_dict, name, tag in [
        (gemma_mid,  gemma_contrast,   "Gemma 4 8B",  "gemma_mid"),
        (gemma_late, gemma_contrast,   "Gemma 4 8B",  "gemma_late"),
        (ap_mid,     apertus_contrast, "Apertus 8B",  "apertus_mid"),
        (ap_late,    apertus_contrast, "Apertus 8B",  "apertus_late"),
    ]:
        if len(layers_subset) < 2:
            print(f"  skipping {tag}: fewer than 2 layers available")
            continue
        cka = compute_cka_matrix(contrast_dict, layers_subset)
        fig_cka_heatmap(cka, layers_subset, name,
                        args.output_dir / f"fig_cka_{tag}.pdf")

    print("\nComputing valence direction similarity matrices ...")
    fig_valence_direction_sim(
        gemma_contrast, gemma_emotions, circumplex, gemma_layers,
        "Gemma 4 8B",
        args.output_dir / "fig_valdir_sim_gemma.pdf",
    )
    fig_valence_direction_sim(
        apertus_contrast, apertus_emotions, circumplex, apertus_layers,
        "Apertus 8B",
        args.output_dir / "fig_valdir_sim_apertus.pdf",
    )
    print("\nGenerating PCA scatter plots ...")
    if GEMMA_BEST_LAYER in gemma_layers:
        fig_pca_scatter(
            gemma_pca[GEMMA_BEST_LAYER], gemma_emotions,
            GEMMA_BEST_LAYER, "Gemma 4 8B",
            args.output_dir / f"fig_pca_gemma_l{GEMMA_BEST_LAYER}.pdf",
        )
    else:
        print(f"  WARNING: Gemma L{GEMMA_BEST_LAYER} not available, skipping PCA scatter")

    if APERTUS_BEST_LAYER in apertus_layers:
        fig_pca_scatter(
            apertus_pca[APERTUS_BEST_LAYER], apertus_emotions,
            APERTUS_BEST_LAYER, "Apertus 8B",
            args.output_dir / f"fig_pca_apertus_l{APERTUS_BEST_LAYER}.pdf",
        )
    else:
        print(f"  WARNING: Apertus L{APERTUS_BEST_LAYER} not available, skipping PCA scatter")

    summary = {
        "gemma_layers":   gemma_layers,
        "apertus_layers": apertus_layers,
        "gemma_pca": {
            str(l): {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in gemma_pca[l].items()}
            for l in gemma_layers
        },
        "apertus_pca": {
            str(l): {k: v.tolist() if isinstance(v, np.ndarray) else v
                     for k, v in apertus_pca[l].items()}
            for l in apertus_layers
        },
    }
    with open(args.output_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {args.output_dir / 'results_summary.json'}")

    print("\n" + "=" * 60)
    print("PCA SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<12} {'Layer':>6} {'PC1↔val r':>12} {'PC2↔aro r':>12}")
    print(f"  {'-'*44}")
    for l in gemma_layers:
        p = gemma_pca[l]
        print(f"  {'Gemma':<12} {l:>6}  {p['r_val']:>11.3f}  {p['r_aro']:>11.3f}")
    for l in apertus_layers:
        p = apertus_pca[l]
        print(f"  {'Apertus':<12} {l:>6}  {p['r_val']:>11.3f}  {p['r_aro']:>11.3f}")
    print("=" * 60)
    print(f"\nAll figures → {args.output_dir}")


if __name__ == "__main__":
    main()
