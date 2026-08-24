#!/usr/bin/env python3
"""
Apply PCA-based confound mitigation to emotion vectors, following Sofroniew (2026)

Input:  {vectors_dir}/<emotion>/layer_{L}_resid.npy           shape (d_model,)
        {neutral_basis_dir}/layer_{L}_neutral_basis.npy       shape (n_paragraphs, d_model)
Output: {vectors_dir}/<emotion>/layer_{L}_resid_projected.npy shape (d_model,)

Usage:
    python apply_confound_mitigation.py \
        --vectors-dir output_apertus/emotion_vectors \
        --layers 12 16 18 20 22 24 26 28 30
"""

import argparse
from pathlib import Path

import numpy as np


def _neutral_components(basis: np.ndarray, variance_threshold: float) -> np.ndarray:
    """
    PCA on the neutral activation basis (n_paragraphs, d_model). Returns the
    top components whose summed explained variance reaches
    variance_threshold of 50% (like the original paper), shape (n_components, d_model).
    """
    centered = basis - basis.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)     # SVD gives components in Vt (n_paragraphs, d_model)

    variance = s ** 2
    total = variance.sum()
    if total == 0:
        return np.zeros((0, basis.shape[1]), dtype=basis.dtype)

    explained_ratio = np.cumsum(variance) / total #cumulative variance
    n_components = int(np.searchsorted(explained_ratio, variance_threshold) + 1)
    n_components = min(n_components, vt.shape[0])

    return vt[:n_components]


def _project_out(vector: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Remove the span of components from vector."""
    if components.shape[0] == 0:
        return vector
    coeffs = components @ vector          # (n_components,)
    return vector - coeffs @ components


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project neutral-baseline PCA components out of emotion vectors"
    )
    parser.add_argument(
        "--vectors-dir", type=Path, required=True,
        help="folder with <emotion>/layer_{L}_resid.npy, as written by extract_emotion_vectors.py",
    )
    parser.add_argument(
        "--neutral-basis-dir", type=Path, default=None,
        help="folder with layer_{L}_neutral_basis.npy, as written by extract_neutral_baseline.py "
             "(default: '<vectors-dir's parent>/neutral_basis')",
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--variance-threshold", type=float, default=0.5,
        help="cumulative variance explained by the neutral PCA components to remove (default: 0.5)",
    )
    args = parser.parse_args()

    # load some dirs 
    neutral_basis_dir = args.neutral_basis_dir or (args.vectors_dir.parent / "neutral_basis")

    emotion_dirs = sorted(d for d in args.vectors_dir.iterdir() if d.is_dir())
    if not emotion_dirs:
        raise ValueError(f"No emotion subdirectories found in {args.vectors_dir}")

    for layer in args.layers:
        missing = [
            emo_dir.name for emo_dir in emotion_dirs
            if not (emo_dir / f"layer_{layer}_resid.npy").exists()
        ]
        if missing: # add error message so we know whats wrong 
            raise FileNotFoundError(
                f"Layer {layer}: missing layer_{layer}_resid.npy for {len(missing)}/"
                f"{len(emotion_dirs)} emotion(s): {', '.join(sorted(missing))}. "
                "Writing partial output would leave some emotions confound-mitigated "
                "and others not, which downstream scripts would silently mix into one "
                "analysis. Re-run extract_emotion_vectors.py for these emotions first, "
                "or drop this layer from --layers."
            )

        # generate components 
        basis_path = neutral_basis_dir / f"layer_{layer}_neutral_basis.npy"
        basis = np.load(basis_path)
        components = _neutral_components(basis, args.variance_threshold)
        print(
            f"Layer {layer}: removing {components.shape[0]} neutral PCA component(s) "
            f"(>= {args.variance_threshold:.0%} of neutral variance)"
        )

        # remove components 
        for emo_dir in emotion_dirs:
            vector = np.load(emo_dir / f"layer_{layer}_resid.npy")
            projected = _project_out(vector, components).astype(np.float32)
            np.save(emo_dir / f"layer_{layer}_resid_projected.npy", projected)

    print(f"\nDone. Wrote layer_{{L}}_resid_projected.npy next to each emotion's layer_{{L}}_resid.npy")


if __name__ == "__main__":
    main()
