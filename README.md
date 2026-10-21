# Efficiency Lab

How small can a bookkeeping model get? Four models, same 74 held-out-vendor invoice lines,
measured on an Apple Silicon laptop with MLX.

| Model | Accuracy | Median ms/line | Peak memory |
|---|---|---|---|
| Base Qwen2.5-0.5B (prompted) | 16.2% | 70 | 1.14 GB |
| LoRA 0.5B fp16 | 93.2% | 65 | 1.13 GB |
| LoRA 0.5B 4-bit | 71.6% | 44 | 0.47 GB |
| Base Qwen2.5-1.5B (prompted) | 39.2% | 195 | 3.25 GB |

## Findings

- A fine-tuned 0.5B model beats a prompted 1.5B model by 54 points, at one third of the memory.
- Post-training 4-bit quantization cost 21.6 points here, far more than expected.
- 74 test rows means roughly +/-6 points of uncertainty. Small gaps are noise.

## Reproduce

    python bench.py

Needs the fused model from https://github.com/ChadSaglam/swiss-bookkeeping-ai
(`fused` and `fused_4bit` folders next to this one).

## Not tested yet

8-bit quantization, LoRA on a 4-bit base (QLoRA), LoRA on the 1.5B model.
