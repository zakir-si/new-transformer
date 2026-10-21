"""Diagnose PyTorch SDPA backend support for native GQA vs explicit K/V expansion."""
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError as exc:
    raise SystemExit(
        "This diagnostic requires a PyTorch version with torch.nn.attention.sdpa_kernel."
    ) from exc

MIB = 1024 ** 2


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_case(q, k, v, kv_heads, mode, backend, warmup, repeats):
    device = q.device
    group = q.size(1) // kv_heads

    def operation():
        if mode == "native_gqa":
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, enable_gqa=True
            )
        k_full = k.repeat_interleave(group, dim=1)
        v_full = v.repeat_interleave(group, dim=1)
        return F.scaled_dot_product_attention(
            q, k_full, v_full, dropout_p=0.0, is_causal=True
        )

    manager = sdpa_kernel(backends=[backend]) if backend is not None else None
    try:
        if manager is None:
            for _ in range(warmup):
                out = operation()
                del out
            sync(device)
            baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            timings = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if device.type == "cuda":
                    start.record()
                    out = operation()
                    end.record()
                else:
                    t0 = time.perf_counter()
                    out = operation()
                    timings.append((time.perf_counter() - t0) * 1000)
                del out
                if device.type == "cuda":
                    # Events are synchronized after the loop.
                    timings.append((start, end))
            sync(device)
            if device.type == "cuda":
                timings = [a.elapsed_time(b) for a, b in timings]
        else:
            with manager:
                for _ in range(warmup):
                    out = operation()
                    del out
                sync(device)
                baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                timings = []
                for _ in range(repeats):
                    if device.type == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        out = operation()
                        end.record()
                        del out
                        timings.append((start, end))
                    else:
                        t0 = time.perf_counter()
                        out = operation()
                        timings.append((time.perf_counter() - t0) * 1000)
                        del out
                sync(device)
                if device.type == "cuda":
                    timings = [a.elapsed_time(b) for a, b in timings]

        peak_increment_mib = None
        if device.type == "cuda":
            peak_increment_mib = max(
                0, torch.cuda.max_memory_allocated(device) - baseline
            ) / MIB
        return {
            "status": "ok",
            "median_ms": statistics.median(timings),
            "mean_ms": statistics.mean(timings),
            "extra_peak_allocated_mib": peak_increment_mib,
        }
    except Exception as exc:
        return {"status": "unsupported_or_failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="128,512")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=4)
    parser.add_argument("--kv-heads", default="1,2")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="sdpa_diagnostic.jsonl")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: select a GPU runtime before running this diagnostic.")
    if args.dim % args.q_heads:
        raise SystemExit("--dim must be divisible by --q-heads")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmup must be >= 0 and repeats >= 1")

    device = torch.device("cuda")
    dtype = torch.float16
    print(json.dumps({
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
        "device_count": torch.cuda.device_count(),
        "flash_attention_available": (
            torch.backends.cuda.is_flash_attention_available()
            if hasattr(torch.backends.cuda, "is_flash_attention_available") else "unknown"
        ),
    }, indent=2))

    backend_names = [
        ("AUTO", None),
        ("FLASH_ATTENTION", getattr(SDPBackend, "FLASH_ATTENTION", None)),
        ("EFFICIENT_ATTENTION", getattr(SDPBackend, "EFFICIENT_ATTENTION", None)),
        ("CUDNN_ATTENTION", getattr(SDPBackend, "CUDNN_ATTENTION", None)),
        ("MATH", getattr(SDPBackend, "MATH", None)),
    ]
    rows = []
    for seq_len in [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]:
        for kv_heads in [int(s.strip()) for s in args.kv_heads.split(",") if s.strip()]:
            if kv_heads < 1 or args.q_heads % kv_heads:
                print(f"Skipping invalid kv_heads={kv_heads}")
                continue
            q = torch.randn(
                args.batch_size, args.q_heads, seq_len, args.dim // args.q_heads,
                device=device, dtype=dtype
            )
            k = torch.randn(
                args.batch_size, kv_heads, seq_len, args.dim // args.q_heads,
                device=device, dtype=dtype
            )
            v = torch.randn_like(k)
            for mode in ["expanded_gqa", "native_gqa"]:
                for backend_name, backend in backend_names:
                    if backend_name != "AUTO" and backend is None:
                        row = {
                            "seq_len": seq_len, "kv_heads": kv_heads, "mode": mode,
                            "backend": backend_name, "status": "unavailable_in_this_torch_build",
                        }
                    else:
                        result = run_case(
                            q, k, v, kv_heads, mode, backend, args.warmup, args.repeats
                        )
                        row = {
                            "seq_len": seq_len, "kv_heads": kv_heads, "mode": mode,
                            "backend": backend_name, **result,
                        }
                    rows.append(row)
                    print(json.dumps(row))
            del q, k, v
            torch.cuda.empty_cache()

    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"WROTE {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
