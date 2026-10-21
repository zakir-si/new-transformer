# Efficient Transformer Research Lab

A reproducible student-scale project investigating **GPU memory and inference efficiency while measuring language-model quality**. Initial implementation focuses on a controlled MHA-vs-GQA comparison and correctness tests. It is a research scaffold, not a claim of a novel architecture.

## Research question

For a small decoder-only Transformer trained under the same data and optimization budget, can grouped-query attention (GQA) reduce KV-cache memory and/or decode latency while keeping held-out language-model perplexity within a predefined tolerance?

### Important distinctions
- **FlashAttention / PyTorch SDPA** changes the implementation of exact softmax attention; it does not remove quadratic attention FLOPs.
- **GQA** shares key/value heads and mainly targets decode-time KV-cache size and memory bandwidth. A newly initialized GQA model is not an apples-to-apples accuracy comparison until it is trained under a controlled recipe.
- **Linear attention** changes the attention mechanism and may change quality; treat it as a separate experiment.
- GPU utilization percentage alone is not a success metric. Record wall-clock latency, throughput, peak memory, and quality.

## Repository layout

- `src/efficient_transformer.py`: small causal attention and decoder model (MHA/GQA).
- `scripts/smoke_test.py`: CPU/GPU correctness and shape checks.
- `scripts/benchmark.py`: quick forward microbenchmark with machine-readable JSON output.
- `tests/`: unit tests.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/benchmark.py --device auto --seq-lens 128,256,512 --batch-size 2 --repeats 5
pytest -q
```

For an NVIDIA T4 in Colab, choose Runtime → Change runtime type → T4 GPU (if available), clone this repository, install dependencies, and run the same commands. GPU availability and runtime type are not guaranteed.

## What the current code does (and does not do)

The benchmark measures attention-module forward latency and CUDA peak allocated/reserved memory when available. It does **not** yet establish language-model accuracy gains or prove end-to-end training speedups. That requires training both model variants on the same corpus and evaluating the same held-out tokens.

The GQA implementation repeats K/V heads to match query heads. This is a clear reference implementation, not a guarantee of an optimized kernel: expansion may reduce memory savings or speed. We will compare it with PyTorch SDPA and only claim a speed win if measured.

## Fair experiment protocol

1. Fix seeds, dataset split, tokenizer, sequence length, batch size, optimizer, training steps, and evaluation tokens.
2. Train baseline MHA and GQA from scratch with the same data and optimization budget; disclose parameter differences.
3. Warm up kernels; synchronize CUDA around timing; run at least 3 repeated measurements.
4. Report perplexity, peak allocated/reserved VRAM, prefill latency, decode tokens/sec, and KV-cache bytes.
5. Run multiple sequence lengths and batch sizes. Record GPU model, PyTorch/CUDA versions, precision, and commit hash.
6. Use mean ± standard deviation and report OOM/failed runs.
7. Initial goals (targets, not promises): ≥30% reduction in the relevant memory measure, ≥20% latency/throughput improvement, and ≤1% relative perplexity degradation. If the baseline does not bottleneck on memory, a memory win may not translate into speed.

## Planned stages

- [x] Implement MHA/GQA reference modules and smoke checks.
- [ ] Validate correctness and benchmark locally/CI.
- [ ] Add a small language-model training/evaluation script.
- [ ] Run a reproducible T4 Colab experiment.
- [ ] Investigate SDPA/FlashAttention availability and linear-attention baseline separately.
- [ ] Analyze whether any proposed adaptive compression mechanism beats known baselines.

## Scientific caution

Combining known techniques is not automatically novel. Any novelty claim requires a focused literature review, a clearly distinct method, and reproducible results against strong baselines.