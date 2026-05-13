"""
train_local.py — Train OthelloMLP locally (no Colab needed).
Works on Windows server with RTX 3060 or any machine with PyTorch.

Usage:
    python train_local.py --data data_l10_overnight_4ch.npz
    python train_local.py --data Dataset/l10/data_l10_v2.npz --epochs 100
"""
import argparse, os, time, math, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ── Config ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",    type=str, default="data_l10_overnight_4ch.npz",
                    help="Path to .npz data file OR directory of .npz files")
parser.add_argument("--epochs",  type=int, default=100)
parser.add_argument("--batch",   type=int, default=2048)
parser.add_argument("--lr",      type=float, default=3e-4)
parser.add_argument("--out",     type=str, default="best_mlp_model_l10.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Load Data ─────────────────────────────────────────────────────────
def load_npz_files(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        print(f"Loading {len(files)} files from {path}/")
    else:
        files = [path]
        print(f"Loading {path}")

    states_list, masks_list, targets_list, outcomes_list = [], [], [], []
    for f in files:
        d = np.load(f)
        states_list.append(d["states"])
        # Support both key naming conventions
        masks_list.append(d["legal_masks"] if "legal_masks" in d else d["masks"])
        targets_list.append(d["policy_targets"] if "policy_targets" in d else d["targets"])
        outcomes_list.append(d["outcomes"])
        print(f"  {os.path.basename(f)}: {d['states'].shape[0]:,} samples, shape={d['states'].shape[1:]}")

    return (np.concatenate(states_list),
            np.concatenate(masks_list),
            np.concatenate(targets_list),
            np.concatenate(outcomes_list))

states, masks, targets, outcomes = load_npz_files(args.data)
print(f"\nTotal : {len(states):,} samples")
print(f"Shape : {states.shape}")
print(f"W/L/D : {(outcomes==1).mean():.1%} / {(outcomes==-1).mean():.1%} / {(outcomes==0).mean():.1%}")

assert states.shape[1] == 4, f"Expected 4 channels, got {states.shape[1]}!"

# ── Dataset ───────────────────────────────────────────────────────────
class OthelloDS(Dataset):
    def __init__(self, s, m, t, o):
        self.s = torch.from_numpy(s).float()
        self.m = torch.from_numpy(m).float()
        self.t = torch.from_numpy(t).long()
        self.o = torch.from_numpy(o).float()
    def __len__(self):  return len(self.s)
    def __getitem__(self, i): return self.s[i], self.m[i], self.t[i], self.o[i]

dataset  = OthelloDS(states, masks, targets, outcomes.astype(np.float32))
val_size = min(50_000, int(0.1 * len(dataset)))
train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size],
                                generator=torch.Generator().manual_seed(42))

n_workers = 4 if device.type == "cuda" else 2
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=n_workers, pin_memory=(device.type=="cuda"))
val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=n_workers, pin_memory=(device.type=="cuda"))
print(f"Train : {len(train_ds):,}  |  Val: {len(val_ds):,}")

# ── Model ─────────────────────────────────────────────────────────────
class OthelloMLP(nn.Module):
    def __init__(self, in_channels=4, hidden=(1024,512,512,256,256,128), dropout=0.1):
        super().__init__()
        layers, prev = [], in_channels * 64
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout)]
            prev = h
        self.backbone    = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, 64)
        self.value_head  = nn.Sequential(nn.Linear(prev,64), nn.ReLU(True), nn.Linear(64,1), nn.Tanh())

    def forward(self, x):
        f = self.backbone(x.view(x.size(0), -1))
        return self.policy_head(f), self.value_head(f).squeeze(1)

model = OthelloMLP(in_channels=4).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ── Training ──────────────────────────────────────────────────────────
EPOCHS       = args.epochs
LR           = args.lr
WARMUP       = 5
LAMBDA_VALUE = 0.5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def warmup_cosine(epoch):
    if epoch < WARMUP: return (epoch + 1) / WARMUP
    progress = (epoch - WARMUP) / max(1, EPOCHS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

best_val_acc = 0.0
print(f"\n{'='*70}")
print(f"  Training: {EPOCHS} epochs | batch={args.batch} | lr={LR} | device={device}")
print(f"{'='*70}")

for epoch in range(1, EPOCHS+1):
    t0 = time.time()

    # Train
    model.train()
    tr_loss = tr_acc = tr_n = 0
    for s, m, tgt, out in train_loader:
        s, m, tgt, out = s.to(device), m.to(device), tgt.to(device), out.to(device)
        optimizer.zero_grad(set_to_none=True)
        policy, value = model(s)
        policy_masked = policy.masked_fill(m <= 0, -1e9)
        p_loss = F.cross_entropy(policy_masked, tgt)
        v_loss = F.mse_loss(value, out)
        loss   = p_loss + LAMBDA_VALUE * v_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_acc  += (policy_masked.argmax(1) == tgt).sum().item()
        tr_loss += loss.item() * len(s)
        tr_n    += len(s)
    scheduler.step()

    # Val
    model.eval()
    va_acc = va_n = 0
    with torch.no_grad():
        for s, m, tgt, _ in val_loader:
            s, m, tgt = s.to(device), m.to(device), tgt.to(device)
            policy, _ = model(s)
            pred = policy.masked_fill(m <= 0, -1e9).argmax(1)
            va_acc += (pred == tgt).sum().item()
            va_n   += len(s)

    ta = tr_acc / tr_n
    va = va_acc / va_n
    elapsed = time.time() - t0
    marker = " ✅" if va > best_val_acc else ""
    print(f"Ep {epoch:3d}/{EPOCHS} | loss={tr_loss/tr_n:.4f} | tr={ta:.4f} | val={va:.4f} | {elapsed:.0f}s{marker}")

    if va > best_val_acc:
        best_val_acc = va
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {"in_channels": 4, "hidden": [1024,512,512,256,256,128], "dropout": 0.1},
            "epoch": epoch, "best_val_acc": best_val_acc,
        }, args.out)

print(f"\n🏆 Best val_acc: {best_val_acc:.4f}")
print(f"   Saved to   : {args.out}")
