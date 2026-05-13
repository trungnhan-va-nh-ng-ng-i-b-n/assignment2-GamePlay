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

def _negamax_ml(board_list, player, depth, alpha, beta, model, device, deadline=None):
    # Time check — abort if deadline exceeded
    if deadline is not None and time.perf_counter() >= deadline:
        s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v = model(s)
        return v.item(), None

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
        v, _ = _negamax_ml(board_list, -player, depth-1, -beta, -alpha, model, device, deadline)
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
        v, _ = _negamax_ml(nb, -player, depth-1, -beta, -alpha, model, device, deadline)
        v = -v
        if v > best_v:
            best_v, best_m = v, move
        alpha = max(alpha, v)
        if alpha >= beta:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
    return best_v, best_m


# ── Neural MCTS (PUCT — AlphaGo Zero style) ─────────────────────────────────
class _MCTSNode:
    __slots__ = ["board","player","move","parent","children","visits","value_sum","prior","is_expanded"]
    def __init__(self, board, player, move=None, parent=None, prior=0.0):
        self.board=board; self.player=player; self.move=move
        self.parent=parent; self.children=[]; self.visits=0
        self.value_sum=0.0; self.prior=prior; self.is_expanded=False

    @property
    def q_value(self): return self.value_sum/self.visits if self.visits>0 else 0.0

    def puct_score(self, c=1.5):
        # Q từ PARENT's perspective = -q_value (zero-sum game)
        # value_sum được tích lũy từ self.player's perspective
        # → parent muốn maximize: -q_value (opponent's loss = our gain)
        p_visits = self.parent.visits if self.parent else 1
        q = -(self.value_sum / self.visits) if self.visits > 0 else 0.0
        return q + c*self.prior*(p_visits**0.5)/(1+self.visits)


def _mcts_expand(node, model, device):
    from core import GameState
    gs = GameState(node.board, node.player)
    legal = gs.get_legal_moves(node.player)
    s = torch.from_numpy(_board_to_tensor(node.board, node.player)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy, value = model(s)
        probs = F.softmax(policy.squeeze(0), dim=0).cpu().numpy()
        v = value.item()
    if not legal:
        opp = gs.get_legal_moves(-node.player)
        if opp:
            node.children.append(_MCTSNode(node.board, -node.player, None, node, 1.0))
    else:
        for move in legal:
            prior = float(probs[move[0]*8+move[1]])
            cb = _apply_move(node.board, move, node.player)
            node.children.append(_MCTSNode(cb, -node.player, move, node, prior))
    node.is_expanded = True
    return v


def _mcts_backprop(node, value):
    while node:
        node.visits+=1; node.value_sum+=value; value=-value; node=node.parent


def run_mcts(board, player, model, device, time_budget=3.0):
    root = _MCTSNode(board, player)
    deadline = time.perf_counter() + time_budget - 0.05
    _mcts_expand(root, model, device)
    if not root.children: return None, 0
    sims = 0
    while time.perf_counter() < deadline:
        # Select
        node = root
        while node.is_expanded and node.children:
            node = max(node.children, key=lambda n: n.puct_score())
        # Expand + Evaluate
        if not node.is_expanded:
            v = _mcts_expand(node, model, device)
        else:
            s = torch.from_numpy(_board_to_tensor(node.board, node.player)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, val = model(s)
            v = val.item()
        # Backprop
        _mcts_backprop(node, v)
        sims += 1
    best = max(root.children, key=lambda n: n.visits)
    return best.move, sims


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

    # Max practical search depth (odd depths win more due to negamax horizon)
    MAX_DEPTH = 9

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

        # Map search_level (1-10) → actual depth, always ODD
        # Odd depth avoids negamax odd-even oscillation:
        #   depth=1→1, 2→3, 3→3, 4→5, 5→5, 6→7, 7→7, 8→7, 9→9, 10→9
        raw = max(1, min(search_level, self.MAX_DEPTH))
        self.depth = raw if raw % 2 == 1 else raw - 1

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

        # Compile model with TorchScript for 2-5x faster inference
        try:
            dummy = torch.zeros(1, 4, 8, 8).to(self.device)
            self.model = torch.jit.trace(self.model, dummy)
            self.model.eval()
        except Exception:
            pass  # fallback to normal model if trace fails

        val_acc = ckpt.get("best_val_acc", "?")
        epoch   = ckpt.get("epoch", "?")
        print(f"[MLPAgent] Loaded {checkpoint_path}  "
              f"mode={mode}  depth={self.depth}  val_acc={val_acc:.4f}  epoch={epoch}")

    def get_move(self, game_state, remain_time: float = 3.0):
        """remain_time: total time budget (seconds) for this move, same as HC SearchAgent."""
        start = time.perf_counter()
        deadline = start + max(0.1, remain_time - 0.05)  # 0.05s safety margin

        legal = game_state.get_legal_moves(self.color)
        if not legal:
            return None, self._metrics(start, 0, 0.0)

        board = game_state.board   # list[list[int]]

        if self.mode == "policy":
            move, score = self._policy_move(board, legal)

        elif self.mode == "1ply":
            move, score = self._oneply_move(board, legal)

        elif self.mode == "mcts":
            move, sims = run_mcts(board, self.color, self.model, self.device,
                                  time_budget=remain_time)
            if move is None: move = legal[0]
            score = sims  # report sim count as score

        else:  # negamax — iterative deepening with time budget
            global _negamax_cache
            _negamax_cache = {}

            best_move, best_score = legal[0], -1e9

            # Iterative deepening: odd depths only (1→3→5→7→9)
            for depth in range(1, self.MAX_DEPTH + 1, 2):
                if time.perf_counter() >= deadline:
                    break  # no time for next depth

                score, move = _negamax_ml(
                    board, self.color, depth, -1e9, 1e9,
                    self.model, self.device, deadline
                )
                if move is not None and time.perf_counter() < deadline:
                    # Only accept result if search completed before deadline
                    best_move, best_score = move, score

            move, score = best_move, best_score
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
