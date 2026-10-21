"""Summarize benchmark JSONL with relative latency and temporary-memory comparisons."""
import argparse
import json
from pathlib import Path


def pct_speedup(baseline_ms, candidate_ms):
    if baseline_ms is None or candidate_ms is None or baseline_ms <= 0 or candidate_ms <= 0:
        return None
    return (baseline_ms / candidate_ms - 1.0) * 100.0


def pct_memory_reduction(baseline_mib, candidate_mib):
    if baseline_mib is None or candidate_mib is None or baseline_mib <= 0:
        return None
    return (baseline_mib - candidate_mib) / baseline_mib * 100.0


def fmt(value, suffix=""):
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="benchmark_results.jsonl")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Invalid JSON on line {line_no}: {exc}") from exc
    if not rows:
        raise SystemExit("Input JSONL contains no benchmark rows")

    baseline_by_key = {}
    for row in rows:
        if row.get("kv_heads") == row.get("q_heads"):
            key = (
                row.get("device"), row.get("seq_len"), row.get("batch_size"),
                row.get("dim"), row.get("q_heads"), row.get("dtype"),
            )
            baseline_by_key[key] = row

    print(f"Loaded {len(rows)} records from {path}")
    if not any(str(r.get("device", "")).startswith("cuda") for r in rows):
        print("WARNING: no CUDA run detected; these results do not establish T4 GPU efficiency.")
    print()
    print("seq  kv  implementation                 median ms   vs MHA     extra MiB   memory reduction   params")
    print("-" * 111)
    for row in sorted(rows, key=lambda r: (
        r.get("seq_len", 0), r.get("kv_heads", 0), r.get("gqa_mode_requested", "")
    )):
        key = (
            row.get("device"), row.get("seq_len"), row.get("batch_size"),
            row.get("dim"), row.get("q_heads"), row.get("dtype"),
        )
        baseline = baseline_by_key.get(key)
        if baseline is None:
            speedup = memory_reduction = None
        elif row is baseline:
            speedup, memory_reduction = 0.0, 0.0
        else:
            speedup = pct_speedup(
                baseline.get("latency_ms_median"), row.get("latency_ms_median")
            )
            memory_reduction = pct_memory_reduction(
                baseline.get("additional_peak_allocated_mib"),
                row.get("additional_peak_allocated_mib"),
            )

        impl = row.get("implementation_used", row.get("gqa_mode_requested", "unknown"))
        print(
            f"{row.get('seq_len', '-'):>4} "
            f"{row.get('kv_heads', '-'):>3}  "
            f"{impl:<29.29} "
            f"{fmt(row.get('latency_ms_median')):>10} "
            f"{fmt(speedup, '%'):>10} "
            f"{fmt(row.get('additional_peak_allocated_mib')):>11} "
            f"{fmt(memory_reduction, '%'):>18} "
            f"{row.get('parameters', 'n/a'):>8}"
        )

    print()
    print("Positive 'vs MHA' means lower median latency than MHA; positive memory reduction means fewer")
    print("incremental peak allocated MiB than MHA. 'n/a' means the metric is unavailable or the baseline is zero.")
    print("These are attention-forward results, not decode/KV-cache benchmarks; compare multiple full runs.")


if __name__ == "__main__":
    main()
