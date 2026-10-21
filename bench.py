import csv, gc, json, re, statistics, sys, time
import mlx.core as mx
from mlx_lm import load, generate

TEST = "../swiss-bookkeeping-ai/data_hard/test.jsonl"
MODELS = {
    "base 0.5B (prompted)": "Qwen/Qwen2.5-0.5B-Instruct",
    "LoRA 0.5B fp16": "../swiss-bookkeeping-ai/fused",
    "LoRA 0.5B 4-bit": "../swiss-bookkeeping-ai/fused_4bit",
    "base 1.5B (prompted)": "Qwen/Qwen2.5-1.5B-Instruct",
}
only = sys.argv[1:]
test = [json.loads(l) for l in open(TEST, encoding="utf-8")]


def peak_gb():
    try:
        return mx.get_peak_memory() / 1e9
    except AttributeError:
        return mx.metal.get_peak_memory() / 1e9


def reset_peak():
    try:
        mx.reset_peak_memory()
    except AttributeError:
        mx.metal.reset_peak_memory()


def run(path):
    reset_peak()
    model, tok = load(path)
    hits, times = 0, []
    for ex in test:
        prompt = tok.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, tokenize=False)
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=prompt, max_tokens=8)
        times.append(time.perf_counter() - t0)
        m = re.search(r"\d{4}", out)
        hits += (m.group(0) if m else "none") == ex["messages"][-1]["content"]
    res = dict(accuracy=round(hits / len(test), 3),
               median_ms=round(statistics.median(times) * 1000),
               total_s=round(sum(times), 1),
               peak_gb=round(peak_gb(), 2))
    del model, tok
    gc.collect()
    return res


rows = []
for name, path in MODELS.items():
    if only and name not in only:
        continue
    print("running", name, flush=True)
    rows.append(dict(model=name, **run(path)))

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model", "accuracy", "median_ms", "total_s", "peak_gb"])
    w.writeheader()
    w.writerows(rows)

print("| Model | Accuracy | Median ms/line | Total s | Peak GB |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r['model']} | {r['accuracy']:.1%} | {r['median_ms']} | {r['total_s']} | {r['peak_gb']} |")
