#!/usr/bin/env python3
"""
Extract the neutral activation basis used for PCA-based confound mitigation
(following Sofroniew et al. 2026): run each of the 40 neutral paragraphs
(prompts/neutral_texts.txt) through the model separately, using the same
masked-mean-over-tokens method as extract_emotion_vectors.py, giving one raw
vector per paragraph per layer. Stack the per-paragraph vectors into a
(n_paragraphs, d_model) matrix per layer and save it.

This script only produces the neutral basis matrix. Building the actual
contrast vector -- PCA on this matrix, keep the top components explaining 50%
of variance, project them out of each emotion vector -- happens downstream, in
whichever pipeline consumes these vectors.

Output: {output_dir}/layer_{L}_neutral_basis.npy   shape (n_paragraphs, d_model)

Usage:
    python extract_neutral_baseline.py \
        --model swiss-ai/Apertus-8B-Instruct-2509 \
        --output-dir output_apertus/neutral_basis \
        --layers 12 16 18 20 22 24 26 28 30
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from extract_emotion_vectors import extract_emotion_vectors, DEFAULT_MODEL

DEFAULT_NEUTRAL_TEXTS = Path(__file__).parent / "prompts" / "neutral_texts.txt"


def _load_neutral_paragraphs(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No neutral paragraphs found in {path}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the per-paragraph neutral activation basis for PCA-based confound removal"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--neutral-texts", type=Path, default=DEFAULT_NEUTRAL_TEXTS)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="folder to write layer_{L}_neutral_basis.npy into, next to the emotion_vectors it corresponds to",
    )
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    paragraphs = _load_neutral_paragraphs(args.neutral_texts)
    n = len(paragraphs)
    width = len(str(n - 1))
    labels = [f"neutral_{i:0{width}d}" for i in range(n)]

    # extract_emotion_vectors() reads a JSONL of {"emotion": ..., "stories": [...]}
    # entries and writes {output_dir}/{emotion}/layer_{L}_resid.npy, averaging
    # over every story listed for that "emotion". Feed it one synthetic
    # "emotion" per neutral paragraph (a single story each) so each paragraph
    # gets its own vector, computed exactly the same way as a real emotion vector.
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for label, paragraph in zip(labels, paragraphs):
            f.write(json.dumps({"emotion": label, "stories": [paragraph]}) + "\n")
        stories_path = Path(f.name)

    tmp_dir = args.output_dir / "_per_paragraph"
    try:
        extract_emotion_vectors(
            stories_file=stories_path,
            output_dir=tmp_dir,
            layers=args.layers,
            model_name=args.model,
            device=torch.device(args.device),
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    finally:
        stories_path.unlink(missing_ok=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for layer in args.layers:
        vecs = [np.load(tmp_dir / label / f"layer_{layer}_resid.npy") for label in labels]
        basis = np.stack(vecs, axis=0).astype(np.float32)   # (n_paragraphs, d_model)
        np.save(args.output_dir / f"layer_{layer}_neutral_basis.npy", basis)

    shutil.rmtree(tmp_dir)

    print(f"\nNeutral basis ({n} paragraphs) written to {args.output_dir}")


if __name__ == "__main__":
    main()
