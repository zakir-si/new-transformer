"""Fast correctness checks: python scripts/smoke_test.py"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.efficient_transformer import CausalSelfAttention,TinyDecoderLM

def main():
    torch.manual_seed(7); device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    x=torch.randn(2,16,32,device=device)
    for kv in (4,2,1):
        m=CausalSelfAttention(32,4,kv).to(device).eval()
        with torch.no_grad(): y=m(x)
        assert y.shape==x.shape and torch.isfinite(y).all()
        print(f"attention kv_heads={kv}: shape={tuple(y.shape)}, params={sum(p.numel() for p in m.parameters())}, finite=True")
    model=TinyDecoderLM(64,max_seq_len=32,dim=32,num_layers=2,num_q_heads=4,num_kv_heads=2).to(device).eval()
    ids=torch.randint(0,64,(2,12),device=device)
    with torch.no_grad(): logits=model(ids); generated=model.generate(ids[:,:4],max_new_tokens=3)
    assert logits.shape==(2,12,64) and generated.shape==(2,7) and torch.isfinite(logits).all()
    print(f"tiny_lm: logits={tuple(logits.shape)}, generated={tuple(generated.shape)}, finite=True")
    print("SMOKE_TEST=PASS")
if __name__=="__main__": main()
