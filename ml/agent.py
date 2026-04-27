from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from Agents import BaseAgent
from ml.encoding import board_to_tensor, index_to_move, legal_moves_mask
from ml.model import PolicyValueNet


class MLAgent(BaseAgent):
    def __init__(
        self,
        color: int,
        checkpoint_path: str,
        name: str | None = None,
        device: str = "auto",
        temperature: float = 0.0,
    ) -> None:
        if name is None:
            name = "MLAgent"
        super().__init__(color=color, name=name)

        self.temperature = max(0.0, float(temperature))
        self.device = self._resolve_device(device)

        checkpoint = torch.load(Path(checkpoint_path), map_location=self.device)
        model_config = checkpoint.get("model_config", {})

        self.model = PolicyValueNet(
            in_channels=int(model_config.get("in_channels", 4)),
            channels=int(model_config.get("channels", 64)),
            num_blocks=int(model_config.get("num_blocks", 4)),
        ).to(self.device)

        if "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict'")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @staticmethod
    def _resolve_device(device_arg: str) -> torch.device:
        if device_arg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_arg)

    def get_move(self, game_state):
        start_time = time.perf_counter()

        legal_moves = game_state.get_legal_moves(self.color)
        if not legal_moves:
            move_time = time.perf_counter() - start_time
            return None, {
                "move_time": move_time,
                "nodes_explored": 1,
                "heuristic_score": 0.0,
                "depth_reached": 0,
                "pruned_branches": 0,
                "difficulty": "ml",
            }

        state_np = board_to_tensor(game_state.board, self.color)
        legal_mask_np = legal_moves_mask(game_state.board, self.color)

        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        legal_mask = torch.from_numpy(legal_mask_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy_logits = policy_logits.masked_fill(legal_mask <= 0, -1e9)

            if self.temperature > 0:
                probs = F.softmax(policy_logits / self.temperature, dim=1)
                move_index = int(torch.multinomial(probs.squeeze(0), num_samples=1).item())
            else:
                move_index = int(torch.argmax(policy_logits, dim=1).item())

        move = index_to_move(move_index)
        if move not in legal_moves:
            move = random.choice(legal_moves)

        move_time = time.perf_counter() - start_time
        return move, {
            "move_time": move_time,
            "nodes_explored": 1,
            "heuristic_score": float(value.item()),
            "depth_reached": 0,
            "pruned_branches": 0,
            "difficulty": "ml",
        }
