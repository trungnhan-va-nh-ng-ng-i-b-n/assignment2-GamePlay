# ML Extension for Othello Project

This folder provides a complete starter pipeline for a machine learning agent:

- `generate_dataset.py`: creates teacher self-play data from `SearchAgent`
- `train.py`: trains a policy-value neural network (PyTorch)
- `agent.py`: `MLAgent` that loads a trained checkpoint and selects legal moves
- `evaluate.py`: headless benchmarking for MLAgent

## 1. Install ML dependencies

From project root:

```bash
pip install -r requirements.txt -r requirements-ml.txt
```

## 2. Generate teacher dataset

```bash
python -m ml.generate_dataset \
  --output ml/data/selfplay_teacher.npz \
  --num-games 300 \
  --min-level 6 \
  --max-level 10 \
  --think-time 0.8 \
  --random-opening-moves 4
```

Outputs:

- dataset: `ml/data/selfplay_teacher.npz`
- metadata: `ml/data/selfplay_teacher.metadata.json`

## 3. Train model

```bash
python -m ml.train \
  --data ml/data/selfplay_teacher.npz \
  --output-dir ml/checkpoints \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate 1e-3
```

Important outputs:

- `ml/checkpoints/best_model.pt`
- `ml/checkpoints/latest_model.pt`
- `ml/checkpoints/training_history.json`

## 4. Evaluate ML agent

Against random:

```bash
python -m ml.evaluate \
  --checkpoint ml/checkpoints/best_model.pt \
  --opponent random \
  --num-games 40
```

Against search agent:

```bash
python -m ml.evaluate \
  --checkpoint ml/checkpoints/best_model.pt \
  --opponent search \
  --search-level 7 \
  --num-games 20
```

## 5. Colab workflow

Recommended flow:

1. Push this repository to GitHub.
2. Open Google Colab with GPU runtime.
3. Clone your repository in Colab.
4. Install dependencies:

```bash
!pip install -r requirements.txt -r requirements-ml.txt
```

5. Run `python -m ml.train ...` in Colab, then download `best_model.pt`.
6. Put checkpoint in `ml/checkpoints/` locally and evaluate with `ml.evaluate`.

## 6. Next improvements

- Add data augmentation (board symmetries)
- Add replay buffer and iterative re-training
- Combine ML value with shallow search at inference time
