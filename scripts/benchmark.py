"""MHA/GQA forward microbenchmark; outputs JSONL, not quality results."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.efficient_transformer import CausalSelfAttention

def sync(d):
    if d.type=="cuda": torch.cuda.synchronize(d)

def run(d,n,b,dim,qh,kvh,warmup,repeats,dtype):
    layer=CausalSelfAttention(dim,qh,kvh).to(device=d,dtype=dtype).eval()
    x=torch.randn(b,n,dim,device=d,dtype=dtype)
    with torch.no_grad():
        for _ in range(warmup): layer(x)
        sync(d)
        if d.type=="cuda": torch.cuda.reset_peak_memory_stats(d)
        times=[]
        for _ in range(repeats):
            sync(d); t=time.perf_counter()
            layer(x); sync(d); times.append((time.perf_counter()-t)*1000)
        alloc=torch.cuda.max_memory_allocated(d)/(1024**2) if d.type=="cuda" else None
        reserv=torch.cuda.max_memory_reserved(d)/(1024**2) if d.type=="cuda" else None
    return {"device":str(d),"seq_len":n,"batch_size":b,"dim":dim,"q_heads":qh,"kv_heads":kvh,
            "dtype":str(dtype),"latency_ms_mean":sum(times)/len(times),"latency_ms_min":min(times),
            "repeats":repeats,"peak_allocated_mib":alloc,"peak_reserved_mib":reserv,
            "output_finite":bool(torch.isfinite(layer(x)).all().item()),
            "note":"reference implementation repeats K/V heads; speed benefits are not guaranteed"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--seq-lens",default="128,256,512"); p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--dim",type=int,default=128); p.add_argument("--q-heads",type=int,default=4)
    p.add_argument("--kv-heads",default="4,2,1"); p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=10); p.add_argument("--output",default="benchmark_results.jsonl")
    a=p.parse_args()
    if a.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    d=torch.device("cuda" if a.device=="cuda" or (a.device=="auto" and torch.cuda.is_available()) else "cpu")
    dtype=torch.float16 if d.type=="cuda" else torch.float32; rows=[]
    for n in map(int,a.seq_lens.split(",")):
        for kv in map(int,a.kv_heads.split(",")):
            if a.q_heads%kv: print(f"Skipping kv_heads={kv}",file=sys.stderr); continue
            row=run(d,n,a.batch_size,a.dim,a.q_heads,kv,a.warmup,a.repeats,dtype); rows.append(row); print(json.dumps(row))
    with open(a.output,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row)+"\n")
    print(f"WROTE {len(rows)} records to {a.output}")
if __name__=="__main__": main()
