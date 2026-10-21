# Efficiency Lab

How small can a bookkeeping model get? Five models, same 74 held-out-vendor invoice lines,
measured on an Apple Silicon laptop with MLX.

| Model | Accuracy | Median ms/line | Peak memory |
|---|---|---|---|
| Base Qwen2.5-0.5B (prompted) | 16.2% | 69 | 1.14 GB |
| LoRA 0.5B fp16 | 93.2% | 71 | 1.13 GB |
| LoRA 0.5B 4-bit | 71.6% | 61 | 0.44 GB |
| LoRA 0.5B 8-bit | 93.2% | 54 | 0.71 GB |
| Base Qwen2.5-1.5B (prompted) | 39.2% | 226 | 3.25 GB |

## Findings

- A fine-tuned 0.5B model beats a prompted 1.5B model by 54 points, at one third of the memory.
- 8-bit quantization kept accuracy identical to fp16 and cut peak memory by about 37%.
- 4-bit quantization (post-training) cost 21.6 points here.
- Latency varies by about 20 ms between runs, so only large speed gaps are meaningful.
- 74 test rows means roughly +/-6 points of uncertainty. Small gaps are noise.

## Reproduce

    python bench.py

Needs the fused models from https://github.com/ChadSaglam/swiss-bookkeeping-ai
(folders fused, fused_4bit and fused_8bit next to this one).

## Not tested yet

LoRA on a 4-bit base (QLoRA), LoRA on the 1.5B model, a harder test set with real invoices.
