"""
ml/mlp_agent.py — MLAgent dùng OthelloMLP (MLP với policy + value heads).
Compatible với framework gốc (BaseAgent / GameState từ agent_minimax/).

Supports 2 modes:
  - "policy"  : policy head argmax → chọn move trực tiếp (nhanh nhất)
  - "negamax" : Negamax depth-N với value head làm leaf evaluator (mạnh nhất)
"""
from __future__ import annotations

import time
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Import BaseAgent từ project gốc ────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Agents import BaseAgent


# ── OthelloMLP (phải match với model đã train) ─────────────────────────
class OthelloMLP(nn.Module):
    def __init__(self, in_channels=3, hidden=(1024,512,512,256,256,128), dropout=0.1):
        super().__init__()
        layers, prev = [], in_channels * 64
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout)]
            prev = h
        self.backbone    = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, 64)
        self.value_head  = nn.Sequential(
            nn.Linear(prev, 64), nn.ReLU(True), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        f = self.backbone(x.view(x.size(0), -1))
        return self.policy_head(f), self.value_head(f).squeeze(1)


# ── Encoding helpers ────────────────────────────────────────────────────
def _legal_mask(board_list: list, player: int) -> np.ndarray:
    """Dùng GameState.get_legal_moves để tạo flat mask (64,)."""
    from core import GameState
    gs = GameState(board_list, player)
    mask = np.zeros(64, dtype=np.float32)
    for r, c in gs.get_legal_moves(player):
        mask[r * 8 + c] = 1.0
    return mask

def _board_to_tensor(board_list: list, player: int) -> np.ndarray:
    """Chuyển board → (4, 8, 8): own, opp, empty, turn_plane — match training encoding."""
    b = np.array(board_list, dtype=np.float32)
    own   = (b == player).astype(np.float32)
    opp   = (b == -player).astype(np.float32)
    empty = (b == 0).astype(np.float32)
    turn  = np.full((8, 8), 1.0 if player == 1 else 0.0, dtype=np.float32)
    return np.stack([own, opp, empty, turn])   # (4,8,8)

def _apply_move(board_list: list, move: tuple, player: int) -> list:
    """Dùng GameState.apply_move để apply move."""
    from core import GameState
    return GameState(board_list, player).apply_move(move, player).board


# ── Negamax với ML value head ───────────────────────────────────────────
_negamax_cache: dict = {}

def _negamax_ml(board_list, player, depth, alpha, beta, model, device):
    from core import GameState
    gs = GameState(board_list, player)
    legal = gs.get_legal_moves(player)
    opp_legal = gs.get_legal_moves(-player)

    if depth == 0 or (not legal and not opp_legal):
        # Leaf: evaluate với value head
        key = (np.array(board_list).tobytes(), player)
        if key not in _negamax_cache:
            s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v = model(s)
            _negamax_cache[key] = v.item()
        return _negamax_cache[key], None

    if not legal:
        v, _ = _negamax_ml(board_list, -player, depth-1, -beta, -alpha, model, device)
        return -v, None

    # ── Policy-guided move ordering (tốt nhất trước → alpha-beta prune tốt hơn ~10x)
    if depth >= 2 and len(legal) > 1:
        s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
        with torch.no_grad():
            p, _ = model(s)
            p = p.squeeze(0).cpu().numpy()  # (64,)
        legal = sorted(legal, key=lambda m: -p[m[0]*8 + m[1]])  # sort giảm dần

    best_v, best_m = -1e9, legal[0]
    for move in legal:
        nb = _apply_move(board_list, move, player)
        v, _ = _negamax_ml(nb, -player, depth-1, -beta, -alpha, model, device)
        v = -v
        if v > best_v:
            best_v, best_m = v, move
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return best_v, best_m


# ── Batch Negamax (GPU-friendly) ────────────────────────────────────────────────────
class _TreeNode:
    """Node trong game tree cho batch evaluation."""
    __slots__ = ["board", "player", "move_from_parent", "children", "value", "is_leaf"]
    def __init__(self, board, player, move_from_parent=None):
        self.board = board
        self.player = player
        self.move_from_parent = move_from_parent
        self.children = []
        self.value = None
        self.is_leaf = False


def _expand_tree(node: _TreeNode, depth: int, max_leaves: int, leaf_list: list):
    """Expand game tree recursively, collect leaf nodes."""
    from core import GameState
    if len(leaf_list) >= max_leaves:
        node.is_leaf = True
        leaf_list.append(node)
        return

    gs = GameState(node.board, node.player)
    legal = gs.get_legal_moves(node.player)
    opp_legal = gs.get_legal_moves(-node.player)

    if depth == 0 or (not legal and not opp_legal):
        node.is_leaf = True
        leaf_list.append(node)
        return

    if not legal:
        # Pass turn
        child = _TreeNode(node.board, -node.player)
        node.children.append((None, -1, child))  # (move, sign, child)
        _expand_tree(child, depth, max_leaves, leaf_list)
        return

    for move in legal:
        nb = _apply_move(node.board, move, node.player)
        child = _TreeNode(nb, -node.player, move)
        node.children.append((move, -1, child))
        _expand_tree(child, depth - 1, max_leaves, leaf_list)


def _backpropagate(node: _TreeNode) -> float:
    """Backpropagate values up the tree (negamax style)."""
    if node.is_leaf:
        return node.value

    best = -1e9
    for move, sign, child in node.children:
        v = sign * _backpropagate(child)
        if v > best:
            best = v
    node.value = best
    return best


def _batch_negamax(board_list, player, depth, model, device, max_leaves=50000):
    """GPU-friendly negamax: expand tree in Python, batch evaluate all leaves."""
    root = _TreeNode(board_list, player)
    leaf_list = []

    _expand_tree(root, depth, max_leaves, leaf_list)

    # Batch evaluate all leaf nodes at once
    tensors = [_board_to_tensor(n.board, n.player) for n in leaf_list]
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with torch.no_grad():
        _, values = model(batch)
    values = values.cpu().numpy()

    for node, val in zip(leaf_list, values):
        node.value = float(val)

    # Backpropagate
    _backpropagate(root)

    # Pick best move at root
    best_v, best_m = -1e9, None
    for move, sign, child in root.children:
        v = sign * (child.value if child.value is not None else 0.0)
        if v > best_v:
            best_v, best_m = v, move

    return best_v, best_m


# ── MLPAgent ────────────────────────────────────────────────────────────
class MLPAgent(BaseAgent):
    """
    MLAgent dùng OthelloMLP (train từ Minimax Level-10 data).

    Modes:
      "policy"  — policy head argmax, không search (cực nhanh)
      "1ply"    — 1-ply batch với value head (nhanh + ổn)
      "negamax" — Negamax depth-N + value head (mạnh nhất, chậm hơn)
    """

    # Sequential negamax + alpha-beta + policy ordering
    # depth 8 ~10s/move (max practical), depth 9+ too slow
    DEPTH_MAP = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:8, 10:8}
    # Batch negamax: can go deeper (GPU batch call at leaves)
    BATCH_DEPTH_MAP = {1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:8, 10:8}

    def __init__(
        self,
        color: int,
        checkpoint_path: str,
        name: str | None = None,
        mode: Literal["policy", "1ply", "negamax", "batch_negamax"] = "negamax",
        search_level: int = 5,
        device: str = "auto",
    ):
        if name is None:
            name = f"MLPAgent-{mode}"
        super().__init__(color=color, name=name)

        self.mode = mode
        if mode == "batch_negamax":
            self.depth = self.BATCH_DEPTH_MAP.get(search_level, 4)
        else:
            self.depth = self.DEPTH_MAP.get(search_level, 3)

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        ckpt = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        cfg  = ckpt.get("model_config", {})
        self.model = OthelloMLP(
            in_channels=cfg.get("in_channels", 3),
            hidden     =cfg.get("hidden",      [1024,512,512,256,256,128]),
            dropout    =cfg.get("dropout",     0.1),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval().to(self.device)

        val_acc = ckpt.get("best_val_acc", "?")
        epoch   = ckpt.get("epoch", "?")
        print(f"[MLPAgent] Loaded {checkpoint_path}  "
              f"mode={mode}  depth={self.depth}  val_acc={val_acc:.4f}  epoch={epoch}")

    def get_move(self, game_state):
        start = time.perf_counter()

        legal = game_state.get_legal_moves(self.color)
        if not legal:
            return None, self._metrics(start, 0, 0.0)

        board = game_state.board   # list[list[int]]

        if self.mode == "policy":
            move, score = self._policy_move(board, legal)

        elif self.mode == "1ply":
            move, score = self._oneply_move(board, legal)

        else:  # negamax or batch_negamax
            if self.mode == "batch_negamax":
                score, move = _batch_negamax(
                    board, self.color, self.depth, self.model, self.device
                )
            else:
                global _negamax_cache
                _negamax_cache = {}
                score, move = _negamax_ml(
                    board, self.color, self.depth, -1e9, 1e9, self.model, self.device
                )
            if move is None:
                move = legal[0]

        return move, self._metrics(start, len(legal), float(score))

    def _policy_move(self, board, legal):
        """Policy head argmax."""
        lmask = _legal_mask(board, self.color)
        s  = torch.from_numpy(_board_to_tensor(board, self.color)).unsqueeze(0).to(self.device)
        mk = torch.from_numpy(lmask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.model(s)
            p = p.masked_fill(mk <= 0, -1e9)
        idx = int(p.argmax(1))
        move = (idx // 8, idx % 8)
        if move not in legal:
            move = random.choice(legal)
        return move, v.item()

    def _oneply_move(self, board, legal):
        """1-ply: batch evaluate all resulting boards với value head."""
        from core import GameState
        boards_batch = []
        for mv in legal:
            nb = GameState(board, self.color).apply_move(mv, self.color).board
            boards_batch.append(_board_to_tensor(nb, -self.color))

        batch = torch.from_numpy(np.stack(boards_batch)).to(self.device)
        with torch.no_grad():
            _, values = self.model(batch)
        values = -values.cpu().numpy()   # negate: opponent's value → our value

        best_idx = int(values.argmax())
        return legal[best_idx], float(values[best_idx])

    @staticmethod
    def _metrics(start, nodes, score):
        return {
            "move_time"       : time.perf_counter() - start,
            "nodes_explored"  : nodes,
            "heuristic_score" : score,
            "depth_reached"   : 0,
            "pruned_branches" : 0,
            "difficulty"      : "ml",
        }
