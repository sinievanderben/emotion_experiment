#!/usr/bin/env python3
"""
Extract emotion vectors from a causal LM (optionally with BatchTopK SAEs).

For each emotion in the JSONL file:
  1. Tokenise stories and run them through the model in batches.
  2. Capture residual-stream activations at each target layer via forward hooks.
  3. Optionally pass activations through the corresponding BatchTopK SAE → sparse feature acts.
  4. Accumulate a masked token-level mean over all stories for that emotion.
  5. Save per-emotion / per-layer .npy files and a combined emotion_vectors.json.

Usage (with SAEs):
    python extract_emotion_vectors.py --sae-config sae_config.json \
        [--model swiss-ai/Apertus-8B-Instruct-2509] \
        [--stories-file ...] [--output-dir ...] \
        [--layers 12 16 17 18 19 20] [--batch-size 4] [--max-length 512]

Usage (residual stream only, no SAEs):
    python extract_emotion_vectors.py \
        --model google/gemma-4-E4B-it \
        --layers 12 16 20 \
        [--stories-file ...] [--output-dir ...]

sae_config.json format:
    {
      "sae_base": "/path/to/sae_apertus",
      "layers": {
        "16": { "run": "batchtopk_k160_layer16_20260412_153003", "checkpoint": "final" },
        "17": { "run": "batchtopk_k160_layer17_20260412_152953", "checkpoint": "checkpoint_step_080000" }
      }
    }
    checkpoint values: "final" (root-level weights), "latest" (newest checkpoint_step_*/),
                       or a specific name like "checkpoint_step_080000".
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL     = "swiss-ai/Apertus-8B-Instruct-2509"
STORIES_FILE      = Path("/users/sinievdben/scratch/personal/emotion_experiment/output/stories.jsonl")
OUTPUT_DIR        = Path("/users/sinievdben/scratch/personal/emotion_experiment/output/emotion_vectors")
SAE_CONFIG_FILE   = Path("/users/sinievdben/scratch/personal/emotion_experiment/sae_config.json")
NEUTRAL_TEXTS_FILE = Path("/users/sinievdben/scratch/personal/emotion_experiment/prompts/neutral_texts.txt")

# Skip first TOKEN_OFFSET tokens when averaging — emotional content builds up
# after the narrative is established (following Anthropic's methodology).
TOKEN_OFFSET = 50


# ─── BatchTopK SAE ────────────────────────────────────────────────────────────

class BatchTopKSAE(nn.Module):
    """
    Inference-only BatchTopK SAE.

    Architecture exactly matches the inner class in
    sae/models.py:BatchTopKSAEWrapper._create_batchtopk_sae so that
    checkpoint state dicts load cleanly.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k_per_sample: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k_per_sample = k_per_sample

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, device=device))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, device=device))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device))
        self.b_dec = nn.Parameter(torch.zeros(d_in,  device=device))
        self.register_buffer(
            "num_batches_not_active",
            torch.zeros(d_sae, dtype=torch.long, device=device),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [N, d_in] float tensor of residual-stream activations.

        Returns:
            (z_pre, z_topk): pre-activation and sparse feature activations,
            both [N, d_sae].  Use z_topk for downstream analysis.
        """
        z_pre = x @ self.W_enc + self.b_enc      # [N, d_sae]
        z = torch.relu(z_pre)
        z_topk = self._apply_batch_topk(z)
        return z_pre, z_topk

    def _apply_batch_topk(self, z: torch.Tensor) -> torch.Tensor:
        """Keep top k_per_sample * batch activations across the whole batch."""
        flat = z.view(-1)
        total = int(min(max(1, z.shape[0] * max(self.k_per_sample, 1)), flat.numel()))
        if total >= flat.numel():
            return z
        topk_vals, topk_idx = torch.topk(flat, total)
        masked = torch.zeros_like(flat)
        masked[topk_idx] = topk_vals
        return masked.view_as(z)


# ─── SAE loading ──────────────────────────────────────────────────────────────

def _resolve_sae_paths(layer: int, sae_cfg: dict) -> Tuple[Path, Path]:
    """
    Resolve sae_batchtopk_state.pt and sae_config.json for `layer` using the
    explicit sae_config.json entries (run + step).

    step values:
      "final"           → {sae_base}/{run}/sae_batchtopk_state.pt
      integer or digit  → {sae_base}/{run}/checkpoint_step_{step:06d}/sae_batchtopk_state.pt
    """
    sae_base = Path(sae_cfg["sae_base"])
    layer_cfg = sae_cfg["layers"][str(layer)]
    run_dir = sae_base / layer_cfg["run"]
    step = layer_cfg["step"]

    if str(step) == "final":
        root = run_dir
    else:
        root = run_dir / f"checkpoint_step_{int(step):06d}"

    state_path  = root / "sae_batchtopk_state.pt"
    config_path = root / "sae_config.json"

    for p in (state_path, config_path):
        if not p.exists():
            raise FileNotFoundError(f"SAE file not found: {p}")

    return state_path, config_path


def load_sae(layer: int, sae_cfg: dict, device: torch.device) -> BatchTopKSAE:
    state_path, config_path = _resolve_sae_paths(layer, sae_cfg)

    with open(config_path) as f:
        cfg = json.load(f)

    sae = BatchTopKSAE(
        d_in=cfg["d_in"],
        d_sae=cfg["d_sae"],
        k_per_sample=cfg["k_per_sample"],
        device=device,
    )
    state = torch.load(state_path, map_location=device, weights_only=True)
    sae.load_state_dict(state)
    sae.eval()

    src = state_path.parent.name
    print(
        f"  layer {layer:2d}: {src}  "
        f"(d_in={cfg['d_in']}, d_sae={cfg['d_sae']}, k={cfg['k_per_sample']})"
    )
    return sae


# ─── Numpy bridge ─────────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy without using torch's numpy bridge.

    torch's bridge breaks when it was compiled against numpy 1.x but numpy 2.x
    is installed.  Going via .tolist() + np.array() avoids the bridge entirely.
    """
    return np.array(t.detach().cpu().tolist(), dtype=np.float32)


# ─── Layer accessor ───────────────────────────────────────────────────────────

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


# ─── Main extraction ──────────────────────────────────────────────────────────

def extract_emotion_vectors(
    stories_file: Path,
    output_dir: Path,
    layers: List[int],
    device: torch.device,
    sae_cfg: dict | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 4,
    max_length: int = 512,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load stories ──────────────────────────────────────────────────────────
    emotions_data: Dict[str, List[str]] = {}
    with open(stories_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("stories"):
                emotions_data.setdefault(entry["emotion"], []).extend(entry["stories"])

    print(
        f"Loaded {len(emotions_data)} emotions "
        f"({sum(len(v) for v in emotions_data.values())} stories total)"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {model_name} ...")
    # Gemma 4's tokenizer_config has extra_special_tokens as a list; older
    # transformers expects a dict and calls .keys() on it.  Patch the base
    # class so both list and dict inputs are accepted.
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

    # ── Infer d_model from the loaded model ───────────────────────────────────
    # Gemma 4 nests the text config; fall back to top-level for other models.
    cfg = model.config
    d_model = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

    # ── Load SAEs (optional) ──────────────────────────────────────────────────
    saes: Dict[int, BatchTopKSAE] = {}
    if sae_cfg is not None:
        print("\nLoading SAEs ...")
        saes = {l: load_sae(l, sae_cfg, device) for l in layers}
    else:
        print("\nNo SAE config provided — extracting residual stream vectors only.")

    # ── Accumulators: emotion → layer → (sum, token count) ──────────────────
    # SAE feature activations (only when SAEs are loaded)
    accum: Dict[str, Dict[int, torch.Tensor]] = {
        emotion: {l: torch.zeros(saes[l].d_sae) for l in layers}
        for emotion in emotions_data
    } if saes else {}
    # Raw residual stream activations
    accum_resid: Dict[str, Dict[int, torch.Tensor]] = {
        emotion: {l: torch.zeros(saes[l].d_in if l in saes else d_model) for l in layers}
        for emotion in emotions_data
    }
    tok_counts: Dict[str, Dict[int, int]] = {
        emotion: {l: 0 for l in layers}
        for emotion in emotions_data
    }

    # ── Process each emotion ──────────────────────────────────────────────────
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

            # Register hooks to capture residual stream after each layer
            captured: Dict[int, torch.Tensor] = {}
            hooks = []

            for layer_idx in layers:
                def _make_hook(idx: int):
                    def _hook(module, inp, out):
                        # Transformer layer output is (hidden_states, ...) or
                        # just hidden_states depending on config.
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

            # Pass activations through SAEs, accumulate masked sums
            mask = inputs["attention_mask"]  # [B, S], 1 = real token

            for layer_idx in layers:
                hidden = captured[layer_idx]           # [B, S, d_model]
                B, S, D = hidden.shape

                # Zero out padding AND first TOKEN_OFFSET tokens before accumulating.
                # Emotional content becomes apparent only after the story is
                # established; early tokens are dominated by narrative framing.
                offset_mask = mask.clone()
                offset_mask[:, :TOKEN_OFFSET] = 0
                om = offset_mask.unsqueeze(-1).float()

                tok_counts[emotion][layer_idx] += int(offset_mask.sum().item())

                # SAE features (only when SAEs are loaded)
                if layer_idx in saes:
                    _, feat_acts = saes[layer_idx].encode(hidden.view(B * S, D))
                    feat_acts = feat_acts.view(B, S, -1) * om  # [B, S, d_sae]
                    accum[emotion][layer_idx] += feat_acts.sum(dim=(0, 1)).cpu()

                # Residual stream (raw, always)
                accum_resid[emotion][layer_idx] += (hidden * om).sum(dim=(0, 1)).cpu()

            done = batch_start + len(batch)
            if done % max(batch_size * 5, 20) == 0 or done == n:
                print(f"  {done}/{n}")

    # ── Compute means, save ───────────────────────────────────────────────────
    print("\nSaving ...")
    emotion_vectors: Dict[str, Dict[str, list]] = {}

    for emotion in emotions_data:
        emotion_vectors[emotion] = {}
        safe = emotion.replace(" ", "_").replace("/", "-")
        emo_dir = output_dir / safe
        emo_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in layers:
            n_tok = tok_counts[emotion][layer_idx]

            # SAE feature vector (only when SAEs are loaded)
            if layer_idx in saes:
                mean_acts = (
                    _to_numpy(accum[emotion][layer_idx] / n_tok)
                    if n_tok > 0
                    else np.zeros(saes[layer_idx].d_sae, dtype=np.float32)
                )
                np.save(emo_dir / f"layer_{layer_idx}.npy", mean_acts)
                emotion_vectors[emotion][str(layer_idx)] = mean_acts.tolist()

            # Residual stream vector
            resid_dim = saes[layer_idx].d_in if layer_idx in saes else d_model
            mean_resid = (
                _to_numpy(accum_resid[emotion][layer_idx] / n_tok)
                if n_tok > 0
                else np.zeros(resid_dim, dtype=np.float32)
            )
            np.save(emo_dir / f"layer_{layer_idx}_resid.npy", mean_resid)
            if not saes:
                # With no SAEs, use residual stream as the primary vector
                emotion_vectors[emotion][str(layer_idx)] = mean_resid.tolist()

    with open(output_dir / "emotion_vectors.json", "w") as f:
        json.dump(emotion_vectors, f)

    print(f"\nDone. All outputs in {output_dir}")
    print(f"  emotion_vectors.json  — combined {len(emotions_data)} emotions × {len(layers)} layers")
    print(f"  <emotion>/layer_<N>.npy  — individual arrays")

    # ── Confound removal (Anthropic methodology) ──────────────────────────────
    # 1. Extract SAE activations on emotionally neutral texts.
    # 2. PCA → top components explaining 50% of variance (topic/style confounds).
    # 3. Project those components out of every emotion vector.
    # Projected vectors are saved as layer_<N>_projected.npy alongside the raw ones.
    if NEUTRAL_TEXTS_FILE.exists():
        print("\nConfound removal ...")
        _remove_confounds(
            emotions_data=emotions_data,
            output_dir=output_dir,
            layers=layers,
            saes=saes,
            d_model=d_model,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    else:
        print(f"\nSkipping confound removal: {NEUTRAL_TEXTS_FILE} not found.")


def _remove_confounds(
    emotions_data: Dict[str, List[str]],
    output_dir: Path,
    layers: List[int],
    saes: Dict[int, "BatchTopKSAE"],
    d_model: int,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> None:
    """
    Extract SAE features and residual-stream activations on neutral texts,
    find confound directions via PCA (separately for each space), project them
    out of the emotion vectors, and save:
      layer_<N>_projected.npy       — SAE features, confound-removed
      layer_<N>_resid_projected.npy — residual stream, confound-removed
    """
    from sklearn.decomposition import PCA

    # Load neutral texts
    with open(NEUTRAL_TEXTS_FILE) as f:
        neutral_texts = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(neutral_texts)} neutral texts")

    # ── Extract neutral activations ───────────────────────────────────────────
    neutral_vecs:       Dict[int, List[np.ndarray]] = {l: [] for l in layers}
    neutral_vecs_resid: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

    for batch_start in range(0, len(neutral_texts), batch_size):
        batch = neutral_texts[batch_start : batch_start + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        captured: Dict[int, torch.Tensor] = {}
        hooks = []
        for layer_idx in layers:
            def _make_hook(idx: int):
                def _hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    captured[idx] = hs.detach().float()
                return _hook
            hooks.append(_get_layer(model, layer_idx).register_forward_hook(_make_hook(layer_idx)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        mask = inputs["attention_mask"]
        offset_mask = mask.clone()
        offset_mask[:, :TOKEN_OFFSET] = 0

        tok_counts_batch = offset_mask.sum(dim=1).float()  # (B,)

        for layer_idx in layers:
            hidden = captured[layer_idx]
            B, S, D = hidden.shape
            om = offset_mask.unsqueeze(-1).float()
            hidden_masked = hidden * om

            if layer_idx in saes:
                _, feat_acts = saes[layer_idx].encode(hidden.view(B * S, D))
                feat_acts = feat_acts.view(B, S, -1) * om

            for i in range(B):
                n = tok_counts_batch[i].item()
                if n > 0:
                    if layer_idx in saes:
                        neutral_vecs[layer_idx].append(
                            _to_numpy(feat_acts[i].sum(dim=0) / n)
                        )
                    neutral_vecs_resid[layer_idx].append(
                        _to_numpy(hidden_masked[i].sum(dim=0) / n)
                    )

        done = batch_start + len(batch)
        print(f"  neutral texts: {done}/{len(neutral_texts)}")

    # ── Per-layer PCA + projection (SAE space, only when SAEs are loaded) ────
    for layer_idx in layers:
        if layer_idx not in saes or not neutral_vecs[layer_idx]:
            continue
        mat = np.stack(neutral_vecs[layer_idx], axis=0)   # (n_neutral, d_sae)
        mat = mat - mat.mean(axis=0, keepdims=True)

        pca = PCA().fit(mat)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, 0.50)) + 1
        confound_dirs = pca.components_[:n_components]

        print(f"  layer {layer_idx} [SAE]:   {n_components} confound PCs "
              f"({cumvar[n_components-1]*100:.1f}% neutral variance)")

        for emotion in emotions_data:
            safe = emotion.replace(" ", "_").replace("/", "-")
            vec = np.load(output_dir / safe / f"layer_{layer_idx}.npy").astype(np.float64)
            for pc in confound_dirs:
                pc = pc.astype(np.float64)
                vec = vec - np.dot(vec, pc) * pc
            np.save(output_dir / safe / f"layer_{layer_idx}_projected.npy",
                    vec.astype(np.float32))

        print(f"  layer {layer_idx} [SAE]:   projected vectors saved")

    # ── Per-layer PCA + projection (residual stream) ──────────────────────────
    for layer_idx in layers:
        mat = np.stack(neutral_vecs_resid[layer_idx], axis=0)  # (n_neutral, d_model)
        mat = mat - mat.mean(axis=0, keepdims=True)

        pca = PCA().fit(mat)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, 0.50)) + 1
        confound_dirs = pca.components_[:n_components]

        print(f"  layer {layer_idx} [resid]: {n_components} confound PCs "
              f"({cumvar[n_components-1]*100:.1f}% neutral variance)")

        for emotion in emotions_data:
            safe = emotion.replace(" ", "_").replace("/", "-")
            vec = np.load(output_dir / safe / f"layer_{layer_idx}_resid.npy").astype(np.float64)
            for pc in confound_dirs:
                pc = pc.astype(np.float64)
                vec = vec - np.dot(vec, pc) * pc
            np.save(output_dir / safe / f"layer_{layer_idx}_resid_projected.npy",
                    vec.astype(np.float32))

        print(f"  layer {layer_idx} [resid]: projected vectors saved")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract emotion vectors via a causal LM (optionally with BatchTopK SAEs)"
    )
    parser.add_argument("--model",        type=str, default=DEFAULT_MODEL)
    parser.add_argument("--sae-config",   type=Path, default=None,
                        help="Path to sae_config.json. Omit to extract residual stream only.")
    parser.add_argument("--stories-file", type=Path, default=STORIES_FILE)
    parser.add_argument("--output-dir",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--layers",       type=int, nargs="+", default=None)
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--max-length",   type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    sae_cfg = None
    if args.sae_config is not None:
        with open(args.sae_config) as f:
            sae_cfg = json.load(f)

    if args.layers is not None:
        layers = args.layers
    elif sae_cfg is not None:
        layers = [int(k) for k in sae_cfg["layers"]]
    else:
        parser.error("--layers is required when --sae-config is not provided")

    extract_emotion_vectors(
        stories_file=args.stories_file,
        output_dir=args.output_dir,
        layers=layers,
        sae_cfg=sae_cfg,
        model_name=args.model,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
