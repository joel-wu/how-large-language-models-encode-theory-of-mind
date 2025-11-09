# How Large Language Models Encode Theory-of-Mind: A Study on Sparse Parameter Patterns

[![NPJAI 2025](https://img.shields.io/badge/NPJAI-2025-green)](https://www.nature.com/articles/s44387-025-00031-9)
[![arXiv](https://img.shields.io/badge/arXiv-2504.04238-red)](https://arxiv.org/abs/2504.04238)

This repository contains the official code for the NPJ AI paper **“How Large Language Models Encode Theory-of-Mind: A Study on Sparse Parameter Patterns.”**

---

1. `create_gradient.py` → compute squared gradients and save as model
2. `chunk_gradient.py` → split gradient into per-layer chunks
3. `ToM_and_perplexity_evaluation.py` → build masked models for each `m`, run ToM & perplexity eval
4. `summarize.py` → aggregate ToM results

Replace every `[]` with your own paths.

---

## 1. Create Gradients

```bash
# TOM dataset (full sequence, last-token only supervision)
python create_gradient.py \
  --model [MODEL_ID] \
  --dataset tom \
  --data_path [/path/to/tom_training_data.json] \
  --nsamples 100 \
  --seqlen 0 \
  --seed 0 \
  --cache_dir [CACHE_DIR] \
  --out [OUT_DIR_FOR_TOM_GRAD]
```

```bash
# C4 dataset (random 128-token windows)
python create_gradient.py \
  --model [MODEL_ID] \
  --dataset c4 \
  --nsamples 100 \
  --seqlen 128 \
  --seed 0 \
  --cache_dir [CACHE_DIR] \
  --out [OUT_DIR_FOR_C4_GRAD]
```

---

## 2. Chunk Gradients

```bash
python chunk_gradient.py \
  --model [OUT_DIR_FOR_TOM_GRAD] \
  --output_path [TOM_CHUNKS_DIR] \
  --cache_dir [CACHE_DIR] \
  --device_map auto
```

```bash
python chunk_gradient.py \
  --model [OUT_DIR_FOR_C4_GRAD] \
  --output_path [C4_CHUNKS_DIR] \
  --cache_dir [CACHE_DIR] \
  --device_map auto
```

---

## 3. ToM + Perplexity Evaluation

```bash
python ToM_and_perplexity_evaluation.py \
  --model [MODEL_ID] \
  --grad_tom_chunks [TOM_CHUNKS_DIR] \
  --grad_c4_chunks [C4_CHUNKS_DIR] \
  --tom_tasks [/path/to/tom_tasks.py] \
  --out_dir [EVAL_OUT_DIR] \
  --cache_dir [CACHE_DIR] \
  --tensor_parallel_size 1 \
  --max_model_len 1024 \
  --batch_size 64 \
  --reps 5 \
  --m_start 0.0 --m_end 5e-5 --m_step 2e-6
```

---

## 4. Summarize ToM Scores

```bash
python summarize.py \
  --root [EVAL_OUT_DIR]/tom \
  --reps 5 \
  --out_csv [EVAL_OUT_DIR]/tom_summary.csv
```
