#!/usr/bin/env python3
"""
Extract emotion vectors from a causal language model.

For each emotion in the JSONL file:
  1. Tokenise stories and run them through the model in batches.
  2. Capture residual-stream activations at each target layer via forward hooks.
  3. Accumulate a masked token-level mean over all stories for that emotion.
  4. Save per-emotion / per-layer .npy files.

Usage:
    python extract_emotion_vectors.py \
        --model swiss-ai/Apertus-8B-Instruct-2509 \
        --stories-file stories.jsonl \
        --output-dir ./emotion_vectors \
        --layers 12 16 18 20 24
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
STORIES_FILE  = Path(__file__).parent / "output/stories.jsonl"
OUTPUT_DIR    = Path(__file__).parent / "output/emotion_vectors"

# Skip first TOKEN_OFFSET tokens when averaging — emotional content builds up
# after the narrative is established.
TOKEN_OFFSET = 50


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return np.array(t.detach().cpu().tolist(), dtype=np.float32)


def _get_layer(model: nn.Module, idx: int) -> nn.Module:
    """
    Return transformer layer `idx`, handling different model families:
      - LLaMA / Apertus / Gemma 1-3:  model.model.layers[idx]
      - Gemma 4 (multimodal):          model.model.language_model.model.layers[idx]
    """
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
    raise AttributeError(
        f"Cannot locate transformer layer {idx} in {type(model).__name__}. "
        "Add the correct path to _get_layer()."
    )


def _iter_jsonl(path: Path):
    """Yield one parsed JSON object per non-empty line of a JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_emotions_data(
    stories_file: Optional[Path] = None,
    stories_dataset: Optional[str] = None,
    stories_split: str = "train",
) -> Dict[str, List[str]]:
    """
    Build a {emotion: [story, ...]} mapping from the generated stories.

    Provide exactly one source:
      - stories_file:    path to a local stories.jsonl (one prompt per line, each
                         with an "emotion" field and a "stories" list).
      - stories_dataset: a Hugging Face dataset repo id, e.g.
                         "snae/emotion_stories_Apertus_8B_Instruct" or
                         "snae/emotion_stories_gemma_4_4B". Requires the
                         `datasets` package (`pip install datasets`).
    """
    if stories_dataset:
        from datasets import load_dataset  # imported lazily so it stays optional
        print(f"Loading stories from Hugging Face dataset: {stories_dataset} "
              f"(split={stories_split})")
        rows = load_dataset(stories_dataset, split=stories_split)
    elif stories_file:
        print(f"Loading stories from local file: {stories_file}")
        rows = _iter_jsonl(stories_file)
    else:
        raise ValueError("Provide either stories_file or stories_dataset.")

    emotions_data: Dict[str, List[str]] = {}
    for entry in rows:
        if entry.get("stories"):
            emotions_data.setdefault(entry["emotion"], []).extend(entry["stories"])
    return emotions_data


def extract_emotion_vectors(
    output_dir: Path,
    layers: List[int],
    device: torch.device,
    stories_file: Optional[Path] = None,
    stories_dataset: Optional[str] = None,
    stories_split: str = "train",
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 4,
    max_length: int = 512,
    debug_pooled_sim: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    emotions_data = load_emotions_data(
        stories_file=stories_file,
        stories_dataset=stories_dataset,
        stories_split=stories_split,
    )

    print(
        f"Loaded {len(emotions_data)} emotions "
        f"({sum(len(v) for v in emotions_data.values())} stories total)"
    )

    print(f"\nLoading model {model_name} ...")
    # Gemma 4's tokenizer_config has extra_special_tokens as a list; older
    # transformers expects a dict. Patch the base class to accept both.
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    _orig_ssmt = PreTrainedTokenizerBase._set_model_specific_special_tokens
    def _patched_ssmt(self, special_tokens):
        if isinstance(special_tokens, dict):
            _orig_ssmt(self, special_tokens)
    PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched_ssmt

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

    # Gemma 4 nests the text config; fall back to top-level for other models.
    cfg = model.config
    d_model = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

    accum: Dict[str, Dict[int, torch.Tensor]] = {
        emotion: {l: torch.zeros(d_model) for l in layers}
        for emotion in emotions_data
    }
    tok_counts: Dict[str, Dict[int, int]] = {
        emotion: {l: 0 for l in layers}
        for emotion in emotions_data
    }

    is_first_emotion = True
    for emotion, stories in emotions_data.items():
        n = len(stories)
        print(f"\n[{emotion}]  {n} stories")

        for batch_start in range(0, n, batch_size):
            batch = stories[batch_start : batch_start + batch_size]

            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            captured: Dict[int, torch.Tensor] = {}
            hooks = []
            debug_printed = False

            for layer_idx in layers:
                def _make_hook(idx: int):
                    def _hook(module, inp, out):
                        nonlocal debug_printed
                        if not debug_printed and batch_start == 0 and is_first_emotion:
                            print(f"[DEBUG] Layer {idx} output structure:")
                            print(f"  type(out) = {type(out)}")
                            if isinstance(out, tuple):
                                print(f"  len(out) = {len(out)}")
                                for i in range(min(3, len(out))):
                                    item = out[i]
                                    if hasattr(item, 'shape'):
                                        print(f"    out[{i}].shape = {item.shape}, dtype = {item.dtype}")
                                    else:
                                        print(f"    out[{i}] type = {type(item)}")
                            elif hasattr(out, 'shape'):
                                print(f"  out.shape = {out.shape}, dtype = {out.dtype}")
                            else:
                                print(f"  out has no shape; attributes: {dir(out)[:5]}")
                            if idx == layers[-1]:
                                debug_printed = True

                        hs = out[0] if isinstance(out, tuple) else out
                        captured[idx] = hs.detach().float()
                    return _hook

                hooks.append(
                    _get_layer(model, layer_idx).register_forward_hook(
                        _make_hook(layer_idx)
                    )
                )

            with torch.no_grad():
                model(**inputs)

            for h in hooks:
                h.remove()

            mask = inputs["attention_mask"]

            for layer_idx in layers:
                hidden = captured[layer_idx]           # [B, S, d_model]
                B, S, D = hidden.shape

                # Zero out padding and the first TOKEN_OFFSET tokens before
                # accumulating — early tokens are dominated by narrative framing.
                offset_mask = mask.clone()
                offset_mask[:, :TOKEN_OFFSET] = 0
                om = offset_mask.unsqueeze(-1).float()

                tok_counts[emotion][layer_idx] += int(offset_mask.sum().item())
                accum[emotion][layer_idx] += (hidden * om).sum(dim=(0, 1)).cpu()

            done = batch_start + len(batch)
            if done % max(batch_size * 5, 20) == 0 or done == n:
                print(f"  {done}/{n}")

        is_first_emotion = False

    print("\nSaving ...")
    emotion_vectors: Dict[str, Dict[str, list]] = {}
    first_emotion = next(iter(emotions_data))

    for emotion in emotions_data:
        emotion_vectors[emotion] = {}
        safe = emotion.replace(" ", "_").replace("/", "-")
        emo_dir = output_dir / safe
        emo_dir.mkdir(parents=True, exist_ok=True)

        if debug_pooled_sim and emotion == first_emotion:
            first_pooled: Dict[int, np.ndarray] = {}

        for layer_idx in layers:
            n_tok = tok_counts[emotion][layer_idx]
            mean_vec = (
                _to_numpy(accum[emotion][layer_idx] / n_tok)
                if n_tok > 0
                else np.zeros(d_model, dtype=np.float32)
            )

            if debug_pooled_sim and emotion == first_emotion:
                first_pooled[layer_idx] = mean_vec
                print(f"[DEBUG] first emotion layer {layer_idx} pooled resid norm = "
                      f"{np.linalg.norm(mean_vec):.4f} (tokens={n_tok})")

            np.save(emo_dir / f"layer_{layer_idx}_resid.npy", mean_vec)
            emotion_vectors[emotion][str(layer_idx)] = mean_vec.tolist()

        if debug_pooled_sim and emotion == first_emotion:
            diag_layers = [l for l in [2, 3, 4, 5, 6, 7, 8, 9] if l in first_pooled]
            if len(diag_layers) > 1:
                base_layer = diag_layers[0]
                base_vec = first_pooled[base_layer]
                print("\n[DEBUG] Pooled residual vector similarity diagnostics (first emotion):")
                print(f"  base layer: {base_layer}")
                for layer_idx in diag_layers[1:]:
                    curr_vec = first_pooled[layer_idx]
                    cos_sim = float(
                        np.dot(base_vec, curr_vec) /
                        (np.linalg.norm(base_vec) * np.linalg.norm(curr_vec))
                    )
                    diff_norm = float(np.linalg.norm(base_vec - curr_vec))
                    print(f"  Layer {layer_idx} vs {base_layer}: cos_sim={cos_sim:.6f}, "
                          f"||diff||={diff_norm:.4f}")
                print("[DEBUG] Pooled residual similarity check complete.\n")

    with open(output_dir / "emotion_vectors.json", "w") as f:
        json.dump(emotion_vectors, f)

    print(f"\nDone. All outputs in {output_dir}")
    print(f"  emotion_vectors.json  — combined {len(emotions_data)} emotions × {len(layers)} layers")
    print(f"  <emotion>/layer_<N>_resid.npy  — individual arrays")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract emotion vectors via residual-stream activations"
    )
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--stories-file", type=Path, default=STORIES_FILE,
                        help="Local stories.jsonl. Ignored if --stories-dataset is set.")
    parser.add_argument("--stories-dataset", type=str, default=None,
                        help="Hugging Face dataset repo id to load stories from "
                             "instead of a local file, e.g. "
                             "'snae/emotion_stories_Apertus_8B_Instruct' or "
                             "'snae/emotion_stories_gemma_4_4B'.")
    parser.add_argument("--stories-split", type=str, default="train",
                        help="Split to load when using --stories-dataset (default: train).")
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--layers",       type=int, nargs="+", required=True)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--max-length",   type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--debug-pooled-sim",
        action="store_true",
        help="Print cosine similarities between pooled residual vectors for the first emotion.",
    )
    args = parser.parse_args()

    extract_emotion_vectors(
        stories_file=None if args.stories_dataset else args.stories_file,
        stories_dataset=args.stories_dataset,
        stories_split=args.stories_split,
        output_dir=args.output_dir,
        layers=args.layers,
        model_name=args.model,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        max_length=args.max_length,
        debug_pooled_sim=args.debug_pooled_sim,
    )


if __name__ == "__main__":
    main()
