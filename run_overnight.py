"""
run_overnight.py — Chạy toàn bộ pipeline: gen L10 data + train model
Treo máy đi ngủ, sáng dậy có model mới!

Usage (Windows PowerShell):
    python run_overnight.py

Tuỳ chỉnh:
    WORKERS      = số CPU cores (88 cho server 2696V4)
    GAMES_BATCH  = games mỗi batch (chia nhỏ tránh OOM)
    N_BATCHES    = số batches (tổng games = GAMES_BATCH × N_BATCHES)
    THINK        = think time (0.5s đủ cho L10 reach depth 8)
    LEVEL        = teacher level (10 = strongest)
    DO_TRAIN     = True để train sau khi gen xong
"""

import subprocess, sys, os, time, json, glob
from pathlib import Path

# ══════════════════════════════════════════════
#  CONFIG — chỉnh ở đây
# ══════════════════════════════════════════════
WORKERS     = 88          # CPU threads (88 cho Dual 2696V4)
GAMES_BATCH = 8000        # games / batch  (8000 × 60 ≈ 480k samples / batch)
N_BATCHES   = 5           # 5 batches × 480k ≈ 2.4M samples total
THINK       = 0.5         # think time (0.5s đủ L10 depth 8, nhanh hơn 3.0s)
LEVEL       = 10          # teacher level
OPENING     = 4           # random opening moves
OUT_DIR     = "Dataset/l10"
BASE_SEED   = 42

DO_TRAIN    = True        # True = tự train sau khi gen xong
TRAIN_SCRIPT = "train_only_v2.py"
# ══════════════════════════════════════════════

def run(cmd, label=""):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "✅ OK" if ok else "❌ FAILED"
    print(f"  {status} — {elapsed/60:.1f} min")
    return ok

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"""
╔══════════════════════════════════════════════════╗
  🌙 Overnight Pipeline — L10 Dataset + Training
  Workers     : {WORKERS}
  Games/batch : {GAMES_BATCH}
  N batches   : {N_BATCHES}
  Total games : {GAMES_BATCH * N_BATCHES:,}
  Est samples : ~{GAMES_BATCH * N_BATCHES * 60:,}
  Think time  : {THINK}s
  Teacher     : L{LEVEL}
  Train after : {DO_TRAIN}
  Started     : {time.strftime('%Y-%m-%d %H:%M:%S')}
╚══════════════════════════════════════════════════╝
""")

    total_t0 = time.time()
    completed = 0

    # ── Phase 1: Generate ────────────────────────────────────
    for i in range(1, N_BATCHES + 1):
        out = f"{OUT_DIR}/data_l10_{i}.npz"

        if os.path.exists(out):
            print(f"  ⏩ Batch {i} already exists — skip")
            completed += 1
            continue

        ok = run(
            [sys.executable, "generate_parallel.py",
             "--games",   str(GAMES_BATCH),
             "--workers", str(WORKERS),
             "--level",   str(LEVEL),
             "--think",   str(THINK),
             "--opening", str(OPENING),
             "--seed",    str(BASE_SEED + i * 1000),
             "--output",  out],
            label=f"Batch {i}/{N_BATCHES} → {out}"
        )
        if ok:
            completed += 1

    # ── Phase 1 summary ──────────────────────────────────────
    files = sorted(glob.glob(f"{OUT_DIR}/data_l10_*.npz"))
    print(f"\n{'═'*60}")
    print(f"  Generation done: {completed}/{N_BATCHES} batches")
    print(f"  Files: {len(files)}")

    try:
        import numpy as np
        total = sum(np.load(f)['states'].shape[0] for f in files)
        print(f"  Total samples: {total:,}")
    except Exception as e:
        print(f"  (could not count samples: {e})")

    # ── Phase 2: Train ───────────────────────────────────────
    if DO_TRAIN and completed > 0:
        if not os.path.exists(TRAIN_SCRIPT):
            print(f"\n  ⚠️  {TRAIN_SCRIPT} not found — skip training")
        else:
            run(
                [sys.executable, TRAIN_SCRIPT,
                 "--data-dir", OUT_DIR],
                label=f"Training on {len(files)} L10 files"
            )

    # ── Final summary ────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    print(f"""
╔══════════════════════════════════════════════════╗
  ✅ Pipeline Complete!
  Total time  : {total_elapsed/3600:.1f} hours
  Finished at : {time.strftime('%Y-%m-%d %H:%M:%S')}
╚══════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()
