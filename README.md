# Emotion Vectors in LLMs

Reproduction of [Anthropic's emotion vectors work](https://transformer-circuits.pub/2026/emotions/index.html) on two open-weight models:
- **Apertus 8B** (`swiss-ai/Apertus-8B-Instruct-2509`) — residual stream
- **Gemma 4 E4B** (`google/gemma-4-E4B-it`) — residual stream

The pipeline generates emotion-labeled stories, extracts hidden-state emotion vectors, and analyzes their geometric structure.

The method in this repository follows descriptions given by Anthropic's research and uses the prompts and emotions provided by them. 

---

## Pipeline Overview

```
1. generate_emotion_stories.py      # Generate stories per emotion (Apertus or Gemma)
2. extract_emotion_vectors.py       # Extract activation vectors from model hidden states
3. analyze_emotion_vectors.py       # PCA, UMAP, CKA analysis and figures
4. analyze_cross_model_geometry.py  # Compare emotion geometry across models
5. visualize_token_activations.py   # Token-level projection heatmaps
```

> `steer_emotion_vectors.py` is an experimental prototype (activation steering) that was not used in the paper. See [Experimental](#experimental-steering) below.

Each step has a corresponding SLURM batch script (`run_*.sbatch`) for running on a GPU cluster.

---

## Setup

### Dependencies

```bash
pip install torch transformers>=4.51 numpy scipy scikit-learn matplotlib tqdm pandas umap-learn
```

If you run the code on a cluster, please create an environment where you can install all the required packages. 

> **Gemma 4 note:** Gemma 4 requires `transformers>=4.51`. On some HPC environments this conflicts with preinstalled numpy. Use `install_transformers_new.sh` / `.sbatch` to install a compatible version into a local path.

### Hardcoded Paths

The Python scripts and `.sbatch` files contain absolute paths (e.g. output directories, model cache paths) pointing to an ETH cluster environment. Before running, update the following in each script:

| Script | Variable/Argument to update |
|--------|-----------------------------|
| `extract_emotion_vectors.py` | `--output_dir`, `--stories_file` defaults |
| `steer_emotion_vectors.py` | `SCRIPT_DIR` constant at top of file (experimental, not used in paper) |
| `analyze_emotion_vectors.py` | `--vectors_dir`, `--output_dir` defaults |
| `analyze_cross_model_geometry.py` | `--apertus_dir`, `--gemma_dir`, `--output_dir` defaults |
| `visualize_token_activations.py` | `--output_dir` default |
| `run_*.sbatch` | `--output`, `--error`, model cache paths |

---

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

### 3. Analyze Emotion Vectors

```bash
python analyze_emotion_vectors.py \
    --vectors_dir output_apertus/emotion_vectors \
    --output_dir output_apertus/analysis \
    --nrc_csv emotion_valence_arousal_nrc.csv
```

Produces: cosine similarity heatmap, PCA scatter, UMAP clustering, CKA matrix (PDF figures).

### 4. Cross-Model Geometry Comparison

```bash
python analyze_cross_model_geometry_apstories.py \
    --apertus_dir output_apertus/emotion_vectors \
    --gemma_dir output_gemma/emotion_vectors \
    --output_dir output_cross_model
```

Change the paths in case you want to analyse the results on stories produced by Gemma. 

### 5. Token-Level Visualization

Under construction!!!

```bash
python3 visualize_token_activations.py \
    --sentences_file sentences.json \
    --apertus_vectors output_apertus/emotion_vectors \
    --gemma_vectors output_gemma/emotion_vectors \
    --output_dir output_token_viz
```

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
sbatch slurm/install_transformers_new.sbatch   # once, for Gemma support
sbatch slurm/run_emotion_stories.sbatch
sbatch slurm/run_extract_emotion_vectors.sbatch
sbatch slurm/run_analyze_emotion_vectors.sbatch
```

All jobs request 1–4 A100 GPUs and 32–64 GB RAM. See individual `.sbatch` files for resource requirements.

---

## Experimental: Steering

`steer_emotion_vectors.py` and `slurm/run_steer_emotion_vectors.sbatch` are **not part of the paper** and were not used in the analysis. The script was a rough prototype for causal validation via activation steering (adding a scaled emotion direction to the residual stream during generation and scoring output with the NRC VAD lexicon). It ran once on Apertus and produced output in `output_apertus/steering/`, but was never developed further.

The script requires updating the hardcoded paths at the top before it can be run in a different environment.

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
