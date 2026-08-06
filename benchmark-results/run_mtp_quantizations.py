#!/usr/bin/env python3
"""Run MTP benchmarks for all 3 quantizations sequentially."""
import subprocess, sys, os

base_dir = "/Users/andreborchert/Downloads/NVMAI"
benchmark_script = os.path.join(base_dir, "benchmark-results", "nvmai_benchmark.py")

models = [
    ("4-bit", os.path.join(base_dir, "models", "qwen36.gturbo")),
    ("6-bit", os.path.join(base_dir, "models", "qwen36-6bit.gturbo")),
    ("8-bit", os.path.join(base_dir, "models", "qwen36-8bit.gturbo")),
]

results = {}

for quant_label, model_path in models:
    print(f"\n{'='*110}")
    print(f"STARTING MTP BENCHMARK: {quant_label} model")
    print(f"Model: {model_path}")
    print(f"{'='*110}\n", flush=True)
    
    result = subprocess.run(
        ["python3", benchmark_script, model_path],
        capture_output=False,
        cwd=base_dir
    )
    
    if result.returncode == 0:
        results[quant_label] = "SUCCESS"
    else:
        results[quant_label] = f"FAILED (exit code {result.returncode})"
    
    print(f"\n{'='*110}")
    print(f"{quant_label} MTP BENCHMARK {'COMPLETED' if result.returncode == 0 else 'FAILED'}")
    print(f"{'='*110}\n")

print(f"\n{'='*110}")
print("ALL MTP BENCHMARKS COMPLETE")
print(f"{'='*110}")
for q, status in results.items():
    print(f"  {q:6s} -> {status}")
print(f"{'='*110}\n")