from __future__ import annotations

import argparse
import json
from pathlib import Path

from Agents import RandomAgent, SearchAgent
from ml.agent import MLAgent
from statistic import run_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MLAgent in headless matches")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--num-games", type=int, default=20, help="Number of games to run")
    parser.add_argument(
        "--opponent",
        type=str,
        default="random",
        choices=["random", "search"],
        help="Opponent type",
    )
    parser.add_argument("--search-level", type=int, default=7, help="Search difficulty if opponent=search")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for MLAgent")
    parser.add_argument("--output", type=str, default="", help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()

    ml_factory = lambda color, name: MLAgent(
        color=color,
        checkpoint_path=args.checkpoint,
        name=name,
        temperature=args.temperature,
    )

    if args.opponent == "random":
        opp_factory = lambda color, name: RandomAgent(color=color, name=name)
        opponent_label = "RandomAgent"
    else:
        opp_factory = lambda color, name: SearchAgent(
            color=color,
            difficulty=args.search_level,
            name=f"SearchAgent-L{args.search_level}",
            remain_time=1.5,
        )
        opponent_label = f"SearchAgent-L{args.search_level}"

    results, metrics = run_batch(
        agent1_factory=ml_factory,
        agent2_factory=opp_factory,
        num_games=args.num_games,
    )

    payload = {
        "checkpoint": args.checkpoint,
        "num_games": args.num_games,
        "opponent": opponent_label,
        "temperature": args.temperature,
        "results": results,
        "metrics": metrics,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved report: {output_path}")

    print("\nEvaluation payload:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
