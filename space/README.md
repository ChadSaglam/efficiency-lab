---
title: Efficiency Lab
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
models:
  - ChadSaglam/swiss-bookkeeping-qwen-lora
  - ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit
datasets:
  - ChadSaglam/swiss-sme-bookkeeping-dataset
short_description: Accuracy vs memory vs speed for small bookkeeping models
---

# Efficiency Lab

Accuracy, latency and peak memory for five small bookkeeping models, measured on the same 74 held-out-vendor invoice lines with MLX on Apple Silicon.

- [Fused model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora)
- [8-bit MLX model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit)
- [Dataset](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset)
- [Benchmark code](https://github.com/ChadSaglam/efficiency-lab)

The results come from synthetic data and one laptop. Small accuracy or latency differences should not be treated as a ranking.
