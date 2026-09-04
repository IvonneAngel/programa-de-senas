import pathlib, json, torch, torch.nn as nn
from pathlib import Path
BASE = Path("C:/Users/riemann/Desktop/programa de señas")
DICT = BASE / "app/public/diccionario_grande.txt"
LSM = BASE / "app/public/lsm_label_map.json"
OUT = BASE / "docs/analisis/fase2-entrenamiento-msl-abc/distilled_lm.pt"
OUT.parent.mkdir(parents=True, exist_ok=True)
print(f"[destilado] cargando {DICT} + {LSM}")
words = [w.strip().lower() for w in DICT.read_text(encoding="utf-8", errors="ignore").splitlines() if len(w.strip())>=2][:5000]
import json as js
lsm = list(js.loads(LSM.read_text(encoding="utf-8")).values())
lsm_words = [v["word"].lower() for v in lsm]
vocab = sorted(set(words + lsm_words))
print(f"vocab {len(vocab)} (92k recortado 5k para destilado rápido)")
stoi = {w:i for i,w in enumerate(vocab)}
# modelo pequeño: emb 64, hidden 128, 2 capas -> ~ (5000*64=320k + lstm ~200k) = <5MB
class SmallLM(nn.Module):
    def __init__(self, vocab, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), emb)
        self.lstm = nn.LSTM(emb, hid, 1, batch_first=True)
        self.fc = nn.Linear(hid, len(vocab))
    def forward(self,x):
        e=self.emb(x)
        o,_=self.lstm(e)
        return self.fc(o[:,-1])
model = SmallLM(vocab)
print(f"params {sum(p.numel() for p in model.parameters())} ~ {sum(p.numel() for p in model.parameters())*4/1024/1024:.1f}MB")
# datos sintéticos: bigramas del diccionario
import random, time
pairs = [(vocab[i], vocab[(i+1)%len(vocab)]) for i in range(len(vocab))]
# entreno rápido 1 epoch para demo destilado
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()
model.train()
start=time.time()
for epoch in range(2):
    random.shuffle(pairs)
    loss_sum=0
    for i in range(0, min(2000,len(pairs)), 32):
        batch=pairs[i:i+32]
        x=torch.tensor([[stoi[w] for w,_ in batch]], dtype=torch.long).t()
        y=torch.tensor([stoi[w2] for _,w2 in batch], dtype=torch.long)
        # x shape [32,1] -> need [32,1] for lstm batch_first
        x=x.t()
        # actually x [32,1]
        logits=model(x)
        loss=crit(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum+=loss.item()
    print(f"epoch {epoch} loss {loss_sum/ (2000/32):.3f} elapsed {time.time()-start:.1f}s")
torch.save({"model_state_dict": model.state_dict(), "vocab": vocab, "stoi": stoi}, OUT)
print(f"guardado {OUT} {OUT.stat().st_size/1024:.1f}KB")
