# Efficient Transformer Research Lab

A reproducible student-scale project investigating **GPU memory and inference efficiency while measuring language-model quality**. The first experiment compares multi-head attention (MHA) with grouped-query attention (GQA). This is a research scaffold, not a claim of a novel architecture.

## Research question

For a small decoder-only Transformer trained under the same data and optimization budget, can GQA reduce relevant memory use or improve latency while keeping held-out perplexity within a predefined tolerance?

### Important distinctions

- **FlashAttention / PyTorch SDPA** can compute exact softmax attention using different kernels; it does not remove the quadratic attention FLOP growth.
- **GQA** reduces the number of key/value heads. Its most direct inference benefit is smaller KV-cache storage and potentially less memory bandwidth.
- **GPU utilization percentage alone is not a success metric.** Record latency, throughput, peak memory, and model quality.
- Combining known techniques is not automatically a novel method. Novelty requires a distinct mechanism, a focused literature review, and reproducible results against strong baselines.

## Repository layout

- `src/efficient_transformer.py`: small causal attention and decoder model (MHA/GQA).
- `scripts/smoke_test.py`: basic correctness and shape checks.
- `scripts/benchmark.py`: comparative forward microbenchmark for MHA, native SDPA GQA, and explicit K/V expansion.
- `scripts/analyze_benchmark.py`: summarizes JSONL results relative to the MHA baseline.
- `scripts/train_lm.py`: small character-level language-model experiment with held-out perplexity and training measurements.
- `tests/`: unit tests, including equivalence of native and expanded GQA paths.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
pytest -q
python scripts/benchmark.py --device auto --seq-lens 128,256,512 --batch-size 2 --dim 128 --q-heads 4 --kv-heads 4,2,1 --implementations native,expand --warmup 5 --repeats 30
python scripts/train_lm.py --steps 500 --seq-len 128 --batch-size 16
```

## Stage 2: efficiency experiment

In Google Colab, select **Runtime → Change runtime type → T4 GPU** if available, then clone the repository and run:

```bash
!git clone https://github.com/zakir-si/new-transformer.git
%cd new-transformer
!pip install -r requirements.txt
!python scripts/smoke_test.py
!pytest -q
!python scripts/benchmark.py --device cuda --seq-lens 128,256,512 --batch-size 2 --dim 128 --q-heads 4 --kv-heads 4,2,1 --implementations native,expand --warmup 5 --repeats 30 --output benchmark_results.jsonl
!cat benchmark_results.jsonl
!python scripts/analyze_benchmark.py benchmark_results.jsonl
!python scripts/diagnose_sdpa.py --seq-lens 128,512 --kv-heads 1,2 --warmup 2 --repeats 5 --output sdpa_diagnostic.jsonl
```

The benchmark reports median/mean/standard deviation of forward latency, positions per second, parameter count/size, and peak CUDA allocation. The incremental peak allocation is measured above the model-plus-input baseline. It logs the GPU and PyTorch/CUDA versions.

**Interpretation rules**
1. Compare rows with the same sequence length, batch size, dimension, query-head count, and dtype.
2. Compare MHA with native GQA to see the architecture-level difference; compare native GQA with explicit expansion to see how much the implementation path matters.
3. Use medians and variability, not the single fastest run. Repeat the entire run at least three times if you are preparing a report.
4. The current script measures attention-module forward execution only. It does **not** measure autoregressive decode speed or KV-cache memory; those need a separate incremental-decoding benchmark.
5. A GQA model has fewer projection parameters, so report parameter counts. It is not a fair accuracy comparison until both variants are trained with the same data and optimization budget.

## Research stages

- [x] Stage 1: implement attention/model and basic correctness tests.
- [x] Stage 2 tooling: implement comparative native-GQA vs expanded-GQA microbenchmark and richer metrics.
- [ ] Stage 2 experiment: run on a T4 GPU and save the raw JSONL results.
- [ ] Stage 3: compare held-out perplexity under a controlled training recipe.
- [ ] Stage 4: investigate a distinct mechanism only after strong baselines and a literature review.

## Scientific caution

Current timings are hardware- and backend-dependent. Native GQA support does not guarantee a faster kernel for every sequence length and shape. Do not claim a speedup, memory reduction, or novelty until the relevant experiment has actually run and its result has been checked.
