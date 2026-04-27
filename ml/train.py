from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.dataset import OthelloNpzDataset, split_dataset
from ml.model import PolicyValueNet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _masked_policy_logits(policy_logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    return policy_logits.masked_fill(legal_masks <= 0, -1e9)


def _run_epoch(
    model: PolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    value_loss_weight: float,
):
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_examples = 0
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_accuracy = 0.0

    if len(loader.dataset) == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "policy_acc": 0.0,
        }

    for states, legal_masks, policy_targets, outcomes in loader:
        states = states.to(device, non_blocking=True)
        legal_masks = legal_masks.to(device, non_blocking=True)
        policy_targets = policy_targets.to(device, non_blocking=True)
        outcomes = outcomes.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            policy_logits, values = model(states)
            masked_logits = _masked_policy_logits(policy_logits, legal_masks)

            policy_loss = F.cross_entropy(masked_logits, policy_targets)
            value_loss = F.mse_loss(values, outcomes)
            loss = policy_loss + value_loss_weight * value_loss

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        predictions = torch.argmax(masked_logits, dim=1)
        policy_acc = (predictions == policy_targets).float().mean()

        batch_size = int(states.shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_policy_loss += float(policy_loss.detach().item()) * batch_size
        total_value_loss += float(value_loss.detach().item()) * batch_size
        total_accuracy += float(policy_acc.detach().item()) * batch_size

    return {
        "loss": total_loss / total_examples,
        "policy_loss": total_policy_loss / total_examples,
        "value_loss": total_value_loss / total_examples,
        "policy_acc": total_accuracy / total_examples,
    }


def _save_checkpoint(
    checkpoint_path: Path,
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_metrics: dict,
    val_metrics: dict,
    args,
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "in_channels": int(args.in_channels),
            "channels": int(args.channels),
            "num_blocks": int(args.num_blocks),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_args": vars(args),
    }
    torch.save(payload, checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Othello policy-value model")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--output-dir", type=str, default="ml/checkpoints", help="Directory to save checkpoints")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data}")
    dataset = OthelloNpzDataset(args.data)
    train_dataset, val_dataset = split_dataset(dataset, val_ratio=args.val_ratio, seed=args.seed)

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    model = PolicyValueNet(
        in_channels=args.in_channels,
        channels=args.channels,
        num_blocks=args.num_blocks,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    history = []

    started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            value_loss_weight=args.value_loss_weight,
        )

        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            value_loss_weight=args.value_loss_weight,
        )

        scheduler.step()

        epoch_summary = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['policy_acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['policy_acc']:.4f}"
        )

        latest_path = output_dir / "latest_model.pt"
        _save_checkpoint(latest_path, model, optimizer, epoch, train_metrics, val_metrics, args)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = output_dir / "best_model.pt"
            _save_checkpoint(best_path, model, optimizer, epoch, train_metrics, val_metrics, args)
            print(f"Saved new best checkpoint: {best_path}")

    total_seconds = time.perf_counter() - started_at

    history_path = output_dir / "training_history.json"
    history_payload = {
        "dataset": args.data,
        "num_samples": len(dataset),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
        "total_train_seconds": total_seconds,
        "history": history,
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    print("\nTraining complete")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"History file: {history_path}")
    print(f"Latest checkpoint: {output_dir / 'latest_model.pt'}")
    print(f"Best checkpoint: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
