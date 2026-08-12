# Emotion Vectors in LLMs

Reproduction of [Anthropic's emotion vectors work](https://transformer-circuits.pub/2026/emotions/index.html) on two open-weight models:
- **Apertus 8B** (`swiss-ai/Apertus-8B-Instruct-2509`) — residual stream
- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) — residual stream

The pipeline generates emotion-labeled stories, extracts hidden-state emotion vectors, and analyzes their geometric structure.

The method in this repository follows descriptions given by Anthropic's research and uses the prompts and emotions provided by them.

`extract_emotion_vectors.py` produces the raw per-emotion vector only (token-level
mean activation over that emotion's stories). Confound mitigation — projecting out
the top principal components of neutral-story activations (enough to explain 50% of
variance) from each emotion vector — is a separate step, not performed by this
script. `extract_neutral_baseline.py` extracts the neutral basis needed for that
step: one activation vector per neutral paragraph in `prompts/neutral_texts.txt`
(same masked-mean method), stacked into a `(n_paragraphs, d_model)` matrix per
layer. Whichever pipeline consumes these vectors is responsible for the PCA and
projection itself.

---

## Data Availability (Hugging Face)

The generated emotion stories are published on the Hugging Face Hub, so you can
skip the generation step (Step 1) and go straight to extracting emotion vectors:

| Model | Dataset | Stories |
|-------|---------|---------|
| Apertus 8B | [`snae/emotion_stories_Apertus_8B_Instruct`](https://huggingface.co/datasets/snae/emotion_stories_Apertus_8B_Instruct) | 513 prompts / 1,539 stories |
| Gemma 4 E4B | [`snae/emotion_stories_gemma_4_4B`](https://huggingface.co/datasets/snae/emotion_stories_gemma_4_4B) | 513 prompts / 1,539 stories |

Each dataset is 171 emotions × 3 topics × 3 stories. Load it with:

```python
from datasets import load_dataset

ds = load_dataset("snae/emotion_stories_Apertus_8B_Instruct")
row = ds["train"][0]
print(row["emotion"], "—", row["topic"])
for story in row["stories"]:   # 3 stories per row
    print(story[:80], "...")
```

`extract_emotion_vectors.py` can read these datasets directly via
`--stories-dataset` (see [Step 2](#2-extract-emotion-vectors)).

---

## Pipeline Overview

```
1. generate_emotion_stories.py      # Generate stories per emotion (Apertus or Gemma)
2. extract_emotion_vectors.py       # Extract activation vectors from model hidden states
3. analyze_emotion_vectors.py       # PCA, UMAP, CKA analysis and figures
4. analyze_cross_model_geometry.py  # Compare emotions across models
```

Each step has a corresponding SLURM batch script (`run_*.sbatch`) for running on a GPU cluster.

---

## Setup

### Dependencies

```bash
pip install torch transformers>=4.51 numpy scipy scikit-learn matplotlib tqdm pandas umap-learn
```

Add `datasets` if you want to load the published stories from the Hugging Face
Hub instead of generating them locally (see [Data Availability](#data-availability-hugging-face)):

```bash
pip install datasets
```

If you run the code on a cluster, please create an environment where you can install all the required packages. 

> **Gemma 4 note:** Gemma 4 requires `transformers>=4.51`. On some HPC environments this conflicts with preinstalled numpy. Use `install_transformers_new.sh` / `.sbatch` to install a compatible version into a local path.

### Hardcoded Paths

The Python scripts and `.sbatch` files contain absolute paths (e.g. output directories, model cache paths) pointing to an ETH cluster environment. Before running, update the following in each script:

| Script | Variable/Argument to update |
|--------|-----------------------------|
| `extract_emotion_vectors.py` | `--output-dir`, `--stories-file` defaults |
| `analyze_emotion_vectors.py` | `--vectors-dir`, `--output-dir` defaults |
| `analyze_cross_model_geometry.py` | `--apertus-dir`, `--gemma-dir`, `--output-dir` defaults |
| `run_*.sbatch` | `--output`, `--error`, model cache paths |

---

The difference between `analyze_emotion_vectors.py` and `analyze_cross_model_geometry.py` is that the first script analyses one model at a time and produces visuals for one model, while the other does both at the same time. 

## Step-by-Step Usage

### 1. Generate Emotion Stories

```bash
# Using Apertus 8B
python generate_emotion_stories.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --output_dir output_apertus_stories \
    --all_emotions

# Using Gemma 4 E4B
python generate_emotion_stories.py \
    --model google/gemma-4-E4B-it \
    --output_dir output_gemma_stories \
    --all_emotions
```

Pass `--resume` to continue from a checkpoint if the job was interrupted.

### 2. Extract Emotion Vectors


```bash
# Apertus — residual stream
python extract_emotion_vectors.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --stories-file output_apertus_stories/stories.jsonl \
    --output-dir output_apertus/emotion_vectors \
    --layers 12 16 18 20 22 24 26 28 30

# Gemma — residual stream
python extract_emotion_vectors.py \
    --model google/gemma-4-E4B-it \
    --stories-file output_apertus_stories/stories.jsonl \
    --output-dir output_gemma/emotion_vectors \
    --layers 17 19 27 28 29
```

You can extract emotion vectors using stories from Apertus or from Gemma. Change the path from 

```bash
output_apertus_stories/stories.jsonl
```

to 

```bash
output_gemma_stories/stories.jsonl
```

Please also be aware of changing the name of ```output-dir```. 

**Loading stories from Hugging Face instead of a local file.** Pass
`--stories-dataset` (which overrides `--stories-file`) to pull the published
stories directly from the Hub — no local generation needed:

```bash
# Apertus stories from the Hub
python extract_emotion_vectors.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --stories-dataset snae/emotion_stories_Apertus_8B_Instruct \
    --output-dir output_apertus/emotion_vectors \
    --layers 12 16 18 20 22 24 26 28 30

# Gemma stories from the Hub
python extract_emotion_vectors.py \
    --model google/gemma-4-E4B-it \
    --stories-dataset snae/emotion_stories_gemma_4_4B \
    --output-dir output_gemma/emotion_vectors \
    --layers 17 19 27 28 29
```

This requires the `datasets` package (`pip install datasets`). Use
`--stories-split` to select a split other than the default `train`.

### 3. Analyze Emotion Vectors

```bash
python analyze_emotion_vectors.py \
    --vectors-dir output_apertus/emotion_vectors \
    --output-dir output_apertus/analysis \
    --vad-csv emotion_valence_arousal_nrc.csv
```

Produces: cosine similarity heatmap, PCA scatter, UMAP clustering, CKA matrix (PDF figures).

### 4. Cross-Model Geometry Comparison

```bash
python analyze_cross_model_geometry.py \
    --apertus-dir output_apertus/emotion_vectors \
    --gemma-dir output_gemma/emotion_vectors \
    --output-dir output_cross_model
```

Change the paths in case you want to analyse the results on stories produced by Gemma. 

### 5. Token-Level Visualization

```bash
python3 visualize_token_activations.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --vectors-dir output_apertus/contrast_vectors \
    --layer 24 \
    --sentences sentences.json \
    --output-dir output_token_viz/apertus_l24
```

### 6. Neutral Basis for Confound Mitigation

```bash
python extract_neutral_baseline.py \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --output-dir output_apertus/neutral_basis \
    --layers 12 16 18 20 22 24 26 28 30
```

Runs each of the 40 neutral paragraphs in `prompts/neutral_texts.txt` through
the model separately, using the same masked-mean method as
`extract_emotion_vectors.py`, and stacks them into a `(40, d_model)` matrix per
layer at `output_apertus/neutral_basis/layer_{L}_neutral_basis.npy`. This
script only produces that matrix. To build the true confound-mitigated contrast
vector for an emotion, run PCA on it, keep the top components explaining 50% of
variance, and project those out of the raw emotion vector:

```python
u_e = np.load("output_apertus/emotion_vectors/joy/layer_12_resid.npy")
basis = np.load("output_apertus/neutral_basis/layer_12_neutral_basis.npy")  # (40, d_model)
_, s, vt = np.linalg.svd(basis, full_matrices=False)
k = int(np.searchsorted(np.cumsum(s**2) / (s**2).sum(), 0.5)) + 1
p = vt[:k]                      # (k, d_model), orthonormal
v_e = u_e - (u_e @ p.T) @ p     # confound-mitigated contrast vector
```

Must use the same model and layer set as the emotion vectors it corresponds to.

---

## Cross-Condition Experiments

To test whether emotion geometry is consistent across story generators (i.e., run Apertus on Gemma-generated stories or vice versa), use the dedicated sbatch scripts:

- `run_extract_vectors_apertus_gemstories.sbatch` — Apertus processes Gemma-generated stories
- `run_extract_vectors_gemma_gemstories.sbatch` — Gemma processes its own stories
- `run_analyze_cross_model_gemstories.sbatch` — Compare geometry across conditions

---

## Running on SLURM 

Update the `--output`, `--error`, and path variables in each `.sbatch` file, then submit:

```bash
sbatch slurm/install_transformers_new.sbatch       # once, for Gemma support
sbatch slurm/run_emotion_stories_apertus.sbatch    # or run_emotion_stories_gemma.sbatch
sbatch slurm/run_extract_emotion_vectors_apertus.sbatch  # or the gemma / *_gemstories variants
sbatch slurm/run_extract_neutral_baseline.sbatch   # neutral basis for confound mitigation
python3 analyze_emotion_vectors.py --vectors-dir ... --output-dir ...  # no .sbatch wrapper, fast enough to run directly
```

All jobs request 1–4 A100 GPUs and 32–64 GB RAM. See individual `.sbatch` files for resource requirements.

---

## Attribution

This repository is a research replication of Anthropic's emotion vectors work:

> Sofroniew, Kauvar, Saunders, Chen & et all. (2026). *Emotion Concepts and their Function in a Large Language Model*.  
> https://transformer-circuits.pub/2026/emotions/index.html

The following files originate from that work and are included here solely for
reproducibility purposes:
- `prompts/emotions.txt` — emotion list
- `prompts/story_prompt.txt` — story generation prompt template
- `prompts/topics.txt` — story topics

No license is stated by the original authors. If you are the rights holder and
have concerns about inclusion of these materials, please open an issue.
