"""Train tiny character-level MHA/GQA language models on tiny Shakespeare.

Intended for a first Colab T4 experiment, not a state-of-the-art benchmark.
Each variant gets the same training steps/data generator; report parameter counts
because GQA changes parameter count. Results are exploratory.
"""
import argparse, json, math, random, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from src.efficient_transformer import TinyDecoderLM

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def get_text(path):
    path=Path(path)
    if path.exists(): return path.read_text(encoding="utf-8")
    print(f"Downloading tiny Shakespeare from {DATA_URL}")
    with urllib.request.urlopen(DATA_URL, timeout=30) as response:
        text=response.read().decode("utf-8")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding="utf-8")
    return text

def batch(data, batch_size, seq_len, device):
    starts=torch.randint(0, len(data)-seq_len-1, (batch_size,))
    x=torch.stack([data[i:i+seq_len] for i in starts.tolist()])
    y=torch.stack([data[i+1:i+seq_len+1] for i in starts.tolist()])
    return x.to(device),y.to(device)

@torch.no_grad()
def evaluate(model, data, device, seq_len, batch_size, batches=20):
    model.eval(); losses=[]
    for _ in range(batches):
        x,y=batch(data,batch_size,seq_len,device)
        logits=model(x)
        losses.append(F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1)).item())
    model.train()
    mean=sum(losses)/len(losses)
    return {"loss":mean,"perplexity":math.exp(min(mean,20))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/tinyshakespeare.txt")
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--eval-interval",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--seq-len",type=int,default=128)
    p.add_argument("--dim",type=int,default=128)
    p.add_argument("--layers",type=int,default=2)
    p.add_argument("--q-heads",type=int,default=4)
    p.add_argument("--kv-heads",default="4,2")
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--output",default="training_results.jsonl")
    a=p.parse_args()
    if a.steps<1 or a.seq_len<2 or a.batch_size<1: raise SystemExit("steps, seq-len and batch-size must be positive")
    random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=="cuda": torch.cuda.manual_seed_all(a.seed)
    text=get_text(a.data)
    chars=sorted(set(text)); stoi={ch:i for i,ch in enumerate(chars)}
    encoded=torch.tensor([stoi[c] for c in text],dtype=torch.long)
    split=int(0.9*len(encoded)); train_data=encoded[:split]; val_data=encoded[split:]
    print(json.dumps({"device":str(device),"torch":torch.__version__,"cuda":torch.version.cuda,
                      "text_chars":len(text),"vocab_size":len(chars),"train_tokens":len(train_data),
                      "val_tokens":len(val_data),"steps_per_variant":a.steps}))
    records=[]
    for kv in map(int,a.kv_heads.split(",")):
        if a.q_heads%kv: print(f"skip kv_heads={kv}: must divide q_heads",file=sys.stderr); continue
        # Same initialization seed per variant; random data batches are sampled from same split.
        torch.manual_seed(a.seed)
        model=TinyDecoderLM(len(chars),max_seq_len=a.seq_len,dim=a.dim,num_layers=a.layers,
                            num_q_heads=a.q_heads,num_kv_heads=kv).to(device)
        params=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
        if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter(); model.train(); last_loss=None
        for step in range(1,a.steps+1):
            x,y=batch(train_data,a.batch_size,a.seq_len,device)
            opt.zero_grad(set_to_none=True)
            logits=model(x)
            loss=F.cross_entropy(logits.reshape(-1,len(chars)),y.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            last_loss=loss.item()
            if step==1 or step%a.eval_interval==0 or step==a.steps:
                val=evaluate(model,val_data,device,a.seq_len,a.batch_size,batches=10)
                print(json.dumps({"variant":"MHA" if kv==a.q_heads else f"GQA-{kv}",
                    "step":step,"train_loss":last_loss,**{f"val_{k}":v for k,v in val.items()}}))
        if device.type=="cuda": torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        val=evaluate(model,val_data,device,a.seq_len,a.batch_size,batches=30)
        record={"variant":"MHA" if kv==a.q_heads else f"GQA-{kv}","kv_heads":kv,
                "parameters":params,"steps":a.steps,"train_seconds":elapsed,
                "steps_per_second":a.steps/elapsed,"final_train_loss":last_loss,
                "val_loss":val["loss"],"val_perplexity":val["perplexity"],
                "peak_allocated_mib":torch.cuda.max_memory_allocated(device)/(1024**2) if device.type=="cuda" else None,
                "peak_reserved_mib":torch.cuda.max_memory_reserved(device)/(1024**2) if device.type=="cuda" else None,
                "device":str(device),"seed":a.seed,"seq_len":a.seq_len,"batch_size":a.batch_size}
        records.append(record); print("FINAL "+json.dumps(record))
        del model,opt
        if device.type=="cuda": torch.cuda.empty_cache()
    with open(a.output,"w",encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record)+"\n")
    print(f"WROTE {len(records)} records to {a.output}")

if __name__=="__main__": main()
