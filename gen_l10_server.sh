#!/bin/bash
# gen_l10_server.sh — Chạy gen dataset L10 liên tục trên server 48 core
# Usage: bash gen_l10_server.sh
# Ước tính: ~150 games/h × 48 workers = 7,200 games/h = ~432k samples/h

WORKERS=48
GAMES_PER_BATCH=3000     # ~3000 games/batch × 60 samples ≈ 180k samples/batch
LEVEL=10
THINK=3.0
OPENING=4
OUTPUT_DIR="Dataset/l10"
TOTAL_BATCHES=20         # 20 batches × ~30 min = ~10h (điều chỉnh tuỳ ý)

mkdir -p "$OUTPUT_DIR"
echo "================================================="
echo "  L10 Dataset Generation — $(date)"
echo "  Workers: $WORKERS | Games/batch: $GAMES_PER_BATCH"
echo "  Level: $LEVEL | Think: ${THINK}s | Batches: $TOTAL_BATCHES"
echo "================================================="

TOTAL_SAMPLES=0

for i in $(seq 1 $TOTAL_BATCHES); do
    OUTFILE="${OUTPUT_DIR}/data_l10_${i}.npz"
    SEED=$((42 + i * 1000))
    
    echo ""
    echo "▶ Batch $i/$TOTAL_BATCHES — $(date +%H:%M:%S)"
    
    python3 -u generate_parallel.py \
        --games   $GAMES_PER_BATCH \
        --level   $LEVEL \
        --think   $THINK \
        --workers $WORKERS \
        --seed    $SEED \
        --opening $OPENING \
        --output  "$OUTFILE"
    
    if [ -f "$OUTFILE" ]; then
        SAMPLES=$(python3 -c "import numpy as np; d=np.load('$OUTFILE'); print(len(d['states']))")
        TOTAL_SAMPLES=$((TOTAL_SAMPLES + SAMPLES))
        SIZE=$(du -sh "$OUTFILE" | cut -f1)
        echo "  ✅ Batch $i done: $SAMPLES samples | file=$SIZE | total=$TOTAL_SAMPLES"
    else
        echo "  ❌ Batch $i FAILED — skipping"
    fi
done

echo ""
echo "================================================="
echo "  DONE — $(date)"
echo "  Total batches  : $TOTAL_BATCHES"
echo "  Total samples  : $TOTAL_SAMPLES"
echo "  Output dir     : $OUTPUT_DIR"
echo "  Files          : $(ls $OUTPUT_DIR | wc -l)"
echo "================================================="

# Summary stats
python3 -c "
import glob, numpy as np
files = sorted(glob.glob('${OUTPUT_DIR}/data_l10_*.npz'))
total = sum(np.load(f)['states'].shape[0] for f in files)
print(f'Grand total: {total:,} samples across {len(files)} files')
"
