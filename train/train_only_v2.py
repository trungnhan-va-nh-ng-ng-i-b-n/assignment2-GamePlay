# %% [markdown]
# # 🎮 Othello MLP — Train Only
# Upload `server_5M.npz` → Train → Download `best_mlp_model.pt`

# %%
import os, time, json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% [markdown]
# ## 1. Upload Dataset

# %%
# Mount Google Drive (upload server_5M.npz vào Drive trước!)
from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn file trong Drive (đặt server_5M.npz vào My Drive gốc)
import os
DATASET_PATH = "/content/drive/MyDrive/server_5M.npz"

# Nếu file không ở gốc Drive, sửa đường dẫn ở trên
print(f"Loading from: {DATASET_PATH}")
assert os.path.exists(DATASET_PATH), f"❌ File not found! Upload server_5M.npz vào Google Drive trước."

data     = np.load(DATASET_PATH)
states   = data["states"]          # (N, 4, 8, 8)
masks    = data["legal_masks"]     # (N, 64)
targets  = data["policy_targets"]  # (N,)
outcomes = data["outcomes"]        # (N,)

print(f"Loaded : {len(states):,} samples  shape={states.shape}")
print(f"W/L/D  : {(outcomes==1).mean():.1%} / {(outcomes==-1).mean():.1%} / {(outcomes==0).mean():.1%}")


# %% [markdown]
# ## 2. Dataset & Loader

# %%
class OthelloDS(Dataset):
    def __init__(self, s, m, t, o):
        self.s = torch.from_numpy(s).float()
        self.m = torch.from_numpy(m).float()
        self.t = torch.from_numpy(t).long()
        self.o = torch.from_numpy(o).float()
    def __len__(self):  return len(self.s)
    def __getitem__(self, i): return self.s[i], self.m[i], self.t[i], self.o[i]

dataset  = OthelloDS(states, masks, targets, outcomes.astype(np.float32))
val_size = min(100_000, int(0.1 * len(dataset)))
train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=2048, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

# %% [markdown]
# ## 3. Model

# %%
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

# %% [markdown]
# ## 4. Train

# %%
import math
EPOCHS       = 100   # 100 epochs ~40 phút trên T4, converge tốt với 2.4M data
LR           = 3e-4  # stable hơn với batch 2048
WARMUP       = 5     # warmup 5 epochs đầu
LAMBDA_VALUE = 0.5

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def warmup_cosine(epoch):
    if epoch < WARMUP:
        return (epoch + 1) / WARMUP
    progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)

best_val_acc = 0.0
history = []

for epoch in range(1, EPOCHS+1):
    t0 = time.time()

    # ── Train ──
    model.train()
    tr_loss = tr_ploss = tr_vloss = tr_acc = tr_n = 0
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
        pred = policy_masked.argmax(1)
        tr_acc   += (pred == tgt).sum().item()
        tr_loss  += loss.item() * len(s)
        tr_ploss += p_loss.item() * len(s)
        tr_vloss += v_loss.item() * len(s)
        tr_n     += len(s)
    scheduler.step()

    # ── Val ──
    model.eval()
    va_acc = va_n = 0
    with torch.no_grad():
        for s, m, tgt, out in val_loader:
            s, m, tgt = s.to(device), m.to(device), tgt.to(device)
            policy, _ = model(s)
            pred = policy.masked_fill(m.to(device) <= 0, -1e9).argmax(1)
            va_acc += (pred == tgt).sum().item()
            va_n   += len(s)

    ta = tr_acc / tr_n
    va = va_acc / va_n
    elapsed = time.time() - t0
    print(f"Ep {epoch:2d}/{EPOCHS} | loss={tr_loss/tr_n:.4f} | p={tr_ploss/tr_n:.4f} | v={tr_vloss/tr_n:.4f} | tr_acc={ta:.4f} | val_acc={va:.4f} | {elapsed:.0f}s")

    history.append({"epoch": epoch, "train_acc": ta, "val_acc": va,
                    "loss": tr_loss/tr_n, "p_loss": tr_ploss/tr_n, "v_loss": tr_vloss/tr_n})

    if va > best_val_acc:
        best_val_acc = va
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_config": {"in_channels": 4, "hidden": [1024,512,512,256,256,128], "dropout": 0.1},
            "epoch": epoch,
            "best_val_acc": best_val_acc,
        }, "best_mlp_model.pt")
        print(f"  ✅ Saved best model (val_acc={va:.4f})")

print(f"\n🏆 Best val_acc: {best_val_acc:.4f}")

# %% [markdown]
# ## 5. Training Curves

# %%
epochs_x   = [h["epoch"]     for h in history]
train_accs = [h["train_acc"] for h in history]
val_accs   = [h["val_acc"]   for h in history]
losses     = [h["loss"]      for h in history]
p_losses   = [h["p_loss"]    for h in history]
v_losses   = [h["v_loss"]    for h in history]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Training Curves — Othello MLP (2.4M samples)", fontsize=13, fontweight="bold")

# Plot 1: Policy Accuracy
axes[0].plot(epochs_x, train_accs, label="Train Acc", color="#4CAF50", linewidth=2)
axes[0].plot(epochs_x, val_accs,   label="Val Acc",   color="#2196F3", linewidth=2, linestyle="--")
axes[0].axhline(y=max(val_accs), color="red", linestyle=":", alpha=0.5,
                label=f"Best={max(val_accs):.4f}")
axes[0].set_title("Policy Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0, 1])

# Plot 2: Total Loss
axes[1].plot(epochs_x, losses, color="#FF5722", linewidth=2)
axes[1].set_title("Total Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)

# Plot 3: Policy vs Value Loss
axes[2].plot(epochs_x, p_losses, label="Policy loss", color="#9C27B0", linewidth=2)
axes[2].plot(epochs_x, v_losses, label="Value loss",  color="#FF9800", linewidth=2, linestyle="--")
axes[2].set_title("Policy vs Value Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
plt.show()
print(f"\n🏆 Best val_acc: {max(val_accs):.4f} at epoch {val_accs.index(max(val_accs))+1}")

# %% [markdown]
# ## 6. Download

# %%
files.download("best_mlp_model.pt")
files.download("training_curves.png")
print("Downloaded best_mlp_model.pt + training_curves.png")
