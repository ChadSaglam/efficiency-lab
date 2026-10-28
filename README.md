# Efficiency Lab

A small benchmark of five bookkeeping models, measured on the same 74 held-out-vendor invoice lines with MLX on an Apple Silicon laptop.

## Resources

- [Live results table](https://huggingface.co/spaces/ChadSaglam/efficiency-lab)
- [Fused LoRA model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora)
- [8-bit MLX model](https://huggingface.co/ChadSaglam/swiss-bookkeeping-qwen-lora-mlx-8bit)
- [Dataset and validation scripts](https://huggingface.co/datasets/ChadSaglam/swiss-sme-bookkeeping-dataset)
- [Training and evaluation code](https://github.com/ChadSaglam/swiss-bookkeeping-ai)

## Results

| Model | Accuracy | Median ms/line | Peak memory |
|---|---:|---:|---:|
| Base Qwen2.5-0.5B, prompted | 16.2% | 69 | 1.14 GB |
| LoRA 0.5B fp16 | 93.2% | 71 | 1.13 GB |
| LoRA 0.5B 4-bit | 71.6% | 61 | 0.44 GB |
| LoRA 0.5B 8-bit | 93.2% | 54 | 0.71 GB |
| Base Qwen2.5-1.5B, prompted | 39.2% | 226 | 3.25 GB |

## Findings

- The fine-tuned 0.5B model outperformed the prompted 1.5B model by 54 points while using about one third of the memory.
- The 8-bit variant kept the same accuracy as fp16 and reduced peak memory by about 37%.
- Post-training 4-bit quantization cost 21.6 accuracy points in this run.
- Latency varied by about 20 ms between runs, so only large speed gaps are meaningful.
- With 74 test rows, small accuracy differences are noise; allow roughly ±6 points of uncertainty.

## Method

All variants used the same held-out-vendor test file. Twelve vendors were excluded from training, although invoice phrases can still overlap. The table reports median generation time per line and peak MLX memory on one laptop; it is not a cross-hardware leaderboard.

The data is synthetic and limited to six accounts. These measurements do not estimate accuracy or latency on real bookkeeping workloads.

## Reproduce

Install the Apple Silicon dependencies and run:

```bash
pip install mlx mlx-lm
python bench.py
```

The script expects the fused model folders from [swiss-bookkeeping-ai](https://github.com/ChadSaglam/swiss-bookkeeping-ai) at these relative paths:

```text
../swiss-bookkeeping-ai/fused
../swiss-bookkeeping-ai/fused_4bit
../swiss-bookkeeping-ai/fused_8bit
```

To run selected variants, pass their exact names from `MODELS` in `bench.py`:

```bash
python bench.py "LoRA 0.5B fp16" "LoRA 0.5B 8-bit"
```

Results are written to `results.csv`. The static table in `space/` reads the same CSV format.

## Not tested yet

- LoRA on a 4-bit base, or QLoRA.
- LoRA on the 1.5B model.
- A larger test set with real invoices and independent accounting review.
