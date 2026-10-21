"""Comparative MHA/GQA forward benchmark; emits machine-readable JSONL."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.efficient_transformer import CausalSelfAttention


MIB = 1024 ** 2


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def implementation_used(device, q_heads, kv_heads, requested):
    if q_heads == kv_heads:
        return "mha_sdpa"
    if requested == "native" and device.type == "cuda":
        return "native_gqa_sdpa_requested"
    if requested == "native":
        return "expanded_fallback_non_cuda"
    return "explicit_kv_expansion"


def run(device, seq_len, batch_size, dim, q_heads, kv_heads,
        warmup, repeats, dtype, requested_mode):
    if dim % q_heads:
        raise ValueError("--dim must be divisible by --q-heads")
    layer = CausalSelfAttention(
        dim, q_heads, kv_heads, use_sdpa=True, gqa_mode=requested_mode
    ).to(device=device, dtype=dtype).eval()
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(warmup):
            layer(x)
        sync(device)
        if not torch.isfinite(layer(x)).all().item():
            raise RuntimeError("non-finite output during benchmark preflight")

        baseline_alloc = baseline_reserved = 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            baseline_alloc = torch.cuda.memory_allocated(device)
            baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

        latencies_ms = []
        if device.type == "cuda":
            events = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = layer(x)
                end.record()
                del output
                events.append((start, end))
            torch.cuda.synchronize(device)
            latencies_ms = [start.elapsed_time(end) for start, end in events]
        else:
            for _ in range(repeats):
                start = time.perf_counter()
                output = layer(x)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                del output

        peak_alloc = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None

    median_ms = statistics.median(latencies_ms)
    params = sum(p.numel() for p in layer.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "dim": dim,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "dtype": str(dtype),
        "gqa_mode_requested": requested_mode,
        "implementation_used": implementation_used(device, q_heads, kv_heads, requested_mode),
        "latency_ms_median": median_ms,
        "latency_ms_mean": statistics.mean(latencies_ms),
        "latency_ms_stddev": statistics.pstdev(latencies_ms),
        "latency_ms_min": min(latencies_ms),
        "positions_per_second": (batch_size * seq_len * 1000.0 / median_ms) if median_ms > 0 else None,
        "parameters": params,
        "parameter_size_mib": param_bytes / MIB,
        "baseline_allocated_mib": baseline_alloc / MIB if device.type == "cuda" else None,
        "additional_peak_allocated_mib": max(0, peak_alloc - baseline_alloc) / MIB if device.type == "cuda" else None,
        "peak_allocated_mib": peak_alloc / MIB if peak_alloc is not None else None,
        "additional_peak_reserved_mib": max(0, peak_reserved - baseline_reserved) / MIB if peak_reserved is not None else None,
        "peak_reserved_mib": peak_reserved / MIB if peak_reserved is not None else None,
        "warmup": warmup,
        "repeats": repeats,
        "output_finite": bool(torch.isfinite(layer(x)).all().item()),
        "note": (
            "Forward microbenchmark only; it does not measure autoregressive KV-cache memory. "
            "native GQA speed depends on actual CUDA backend/kernel selection."
        ),
    }
    return result


def parse_int_list(raw, flag):
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{flag} must contain at least one integer")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seq-lens", default="128,256,512")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", default="4,2,1")
    parser.add_argument("--implementations", default="native,expand",
                        help="comma-separated GQA paths: native,expand")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="benchmark_results.jsonl")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.batch_size < 1 or args.dim < 1 or args.q_heads < 1:
        raise SystemExit("batch size, dim, and q-heads must be positive")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmup must be non-negative and repeats must be at least 1")

    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    seq_lens = parse_int_list(args.seq_lens, "--seq-lens")
    kv_heads_list = parse_int_list(args.kv_heads, "--kv-heads")
    modes = [mode.strip() for mode in args.implementations.split(",") if mode.strip()]
    if not modes or any(mode not in {"native", "expand"} for mode in modes):
        raise SystemExit("--implementations must contain native and/or expand")
    if any(n < 1 for n in seq_lens) or any(k < 1 for k in kv_heads_list):
        raise SystemExit("sequence lengths and KV heads must be positive")

    rows = []
    for seq_len in seq_lens:
        for kv_heads in kv_heads_list:
            if args.q_heads % kv_heads:
                print(f"Skipping kv_heads={kv_heads}: q-heads must be divisible by KV heads",
                      file=sys.stderr)
                continue
            # MHA is one baseline; running both GQA modes for MHA would duplicate it.
            run_modes = ["native"] if kv_heads == args.q_heads else modes
            for mode in run_modes:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                row = run(
                    device, seq_len, args.batch_size, args.dim, args.q_heads,
                    kv_heads, args.warmup, args.repeats, dtype, mode
                )
                rows.append(row)
                print(json.dumps(row))

    if not rows:
        raise SystemExit("No valid benchmark configurations were run")
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"WROTE {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
