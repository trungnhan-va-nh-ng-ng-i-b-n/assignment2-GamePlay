"""
ml/mlp_agent.py — Othello ML Agent dùng OthelloMLP (policy + value heads).

Architecture:
    OthelloMLP: 6-layer MLP, input (4, 8, 8) board tensor
        - policy_head: softmax over 64 squares → move prior
        - value_head:  tanh scalar ∈ [-1, 1] → position evaluation

Search algorithm (negamax mode):
    Negamax + alpha-beta pruning, same framework as HC SearchAgent.
    Key differences from HC (hand-crafted heuristic):
        1. Leaf evaluation: value_head(board) replaces hand-crafted formula
        2. Move ordering:   policy_head(board) sorts moves best-first at every
                            interior node (depth ≥ 2), enabling deeper pruning
        3. Iterative deepening over ODD depths (1→3→5→7→9) within time budget
           (odd depths avoid negamax horizon effect on trained value head)
        4. Transposition table: cache leaf evaluations by board+player hash
        5. TorchScript JIT: compiled model for 2-3x faster inference

Modes:
    "policy"  — policy head argmax, no search (fastest, used for quick demos)
    "negamax" — Negamax + alpha-beta + iterative deepening (strongest)

Compatible with BaseAgent / GameState framework from Agents.py / core.py.
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

# ── BaseAgent from project root ─────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Agents import BaseAgent


# ── OthelloMLP ──────────────────────────────────────────────────────────
class OthelloMLP(nn.Module):
    """
    6-layer MLP for Othello position evaluation.

    Input:  (B, 4, 8, 8) board tensor
              ch0: own discs   (1.0 where current player has disc)
              ch1: opp discs   (1.0 where opponent has disc)
              ch2: empty cells (1.0 where cell is empty)
              ch3: turn plane  (1.0 if player==1 (Black), 0.0 if player==-1 (White))

    Output: (policy_logits, value)
              policy_logits: (B, 64) — unnormalised move scores
              value:         (B,)    — tanh position eval ∈ [-1, 1]
                                       +1 = current player wins, -1 = loses
    """
    def __init__(self, in_channels=4, hidden=(1024, 512, 512, 256, 256, 128), dropout=0.1):
        super().__init__()
        layers, prev = [], in_channels * 64
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(True), nn.Dropout(dropout)]
            prev = h
        self.backbone    = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, 64)
        self.value_head  = nn.Sequential(
            nn.Linear(prev, 64), nn.ReLU(True), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x):
        f = self.backbone(x.view(x.size(0), -1))
        return self.policy_head(f), self.value_head(f).squeeze(1)


# ── Board encoding helpers ───────────────────────────────────────────────
def _legal_mask(board_list: list, player: int) -> np.ndarray:
    """Return flat (64,) binary mask of legal moves."""
    from core import GameState
    gs = GameState(board_list, player)
    mask = np.zeros(64, dtype=np.float32)
    for r, c in gs.get_legal_moves(player):
        mask[r * 8 + c] = 1.0
    return mask


def _board_to_tensor(board_list: list, player: int) -> np.ndarray:
    """Convert board → (4, 8, 8) tensor matching training encoding."""
    b = np.array(board_list, dtype=np.float32)
    own   = (b == player).astype(np.float32)
    opp   = (b == -player).astype(np.float32)
    empty = (b == 0).astype(np.float32)
    turn  = np.full((8, 8), 1.0 if player == 1 else 0.0, dtype=np.float32)
    return np.stack([own, opp, empty, turn])  # (4, 8, 8)


def _apply_move(board_list: list, move: tuple, player: int) -> list:
    """Apply move and return new board."""
    from core import GameState
    return GameState(board_list, player).apply_move(move, player).board


# ── Negamax with alpha-beta + ML value head ──────────────────────────────
_negamax_cache: dict = {}   # Transposition table: (board_bytes, player) → value


def _negamax_ml(board_list, player, depth, alpha, beta, model, device, deadline=None):
    """
    Negamax with alpha-beta pruning, using ML model for evaluation.

    Differences from HC negamax:
      - Leaf nodes: evaluated by model.value_head (not hand-crafted heuristic)
      - Interior nodes (depth ≥ 2): moves sorted by model.policy_head priors
        (best-first ordering → more alpha-beta cuts → effectively deeper search)
      - Transposition table: repeated leaf positions reuse cached values

    Args:
        board_list : current board state
        player     : current player (1 = Black, -1 = White)
        depth      : remaining search depth
        alpha, beta: alpha-beta window
        model      : OthelloMLP (JIT compiled)
        device     : torch device
        deadline   : time.perf_counter() deadline — abort if exceeded

    Returns:
        (score, best_move) — score from current player's perspective
    """
    # ── Time guard ──────────────────────────────────────────────────────
    if deadline is not None and time.perf_counter() >= deadline:
        s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, v = model(s)
        return v.item(), None

    from core import GameState
    gs = GameState(board_list, player)
    legal     = gs.get_legal_moves(player)
    opp_legal = gs.get_legal_moves(-player)

    # ── Terminal / leaf node ─────────────────────────────────────────────
    if depth == 0 or (not legal and not opp_legal):
        key = (np.array(board_list).tobytes(), player)
        if key not in _negamax_cache:
            s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, v = model(s)
            _negamax_cache[key] = v.item()
        return _negamax_cache[key], None

    # ── Pass turn ────────────────────────────────────────────────────────
    if not legal:
        v, _ = _negamax_ml(board_list, -player, depth - 1, -beta, -alpha, model, device, deadline)
        return -v, None

    # ── Policy-guided move ordering (best-first → more alpha-beta pruning)
    # Called at every interior node with depth ≥ 2 and multiple moves.
    # The policy head assigns a prior probability to each move;
    # sorting moves by prior probability puts likely-good moves first,
    # which increases the chance of an early alpha-beta cutoff.
    if depth >= 2 and len(legal) > 1:
        s = torch.from_numpy(_board_to_tensor(board_list, player)).unsqueeze(0).to(device)
        with torch.no_grad():
            p, _ = model(s)
            p = p.squeeze(0).cpu().numpy()  # (64,) logits
        legal = sorted(legal, key=lambda m: -p[m[0] * 8 + m[1]])

    # ── Alpha-beta search ────────────────────────────────────────────────
    best_v, best_m = -1e9, legal[0]
    for move in legal:
        nb = _apply_move(board_list, move, player)
        v, _ = _negamax_ml(nb, -player, depth - 1, -beta, -alpha, model, device, deadline)
        v = -v
        if v > best_v:
            best_v, best_m = v, move
        alpha = max(alpha, v)
        if alpha >= beta:
            break  # beta cutoff
        if deadline is not None and time.perf_counter() >= deadline:
            break

    return best_v, best_m


# ── Batch Negamax (GPU-friendly) ─────────────────────────────────────────
class _TreeNode:
    """Node in the game tree for batch leaf evaluation."""
    __slots__ = ["board", "player", "move_from_parent", "children", "value", "is_leaf"]
    def __init__(self, board, player, move_from_parent=None):
        self.board = board; self.player = player
        self.move_from_parent = move_from_parent
        self.children = []; self.value = None; self.is_leaf = False


def _expand_tree(node: _TreeNode, depth: int, max_leaves: int, leaf_list: list):
    """Expand tree recursively, collecting all leaf nodes for batch evaluation."""
    from core import GameState
    if len(leaf_list) >= max_leaves:
        node.is_leaf = True; leaf_list.append(node); return

    gs = GameState(node.board, node.player)
    legal = gs.get_legal_moves(node.player)
    opp_legal = gs.get_legal_moves(-node.player)

    if depth == 0 or (not legal and not opp_legal):
        node.is_leaf = True; leaf_list.append(node); return

    if not legal:
        child = _TreeNode(node.board, -node.player)
        node.children.append((None, -1, child))
        _expand_tree(child, depth, max_leaves, leaf_list)
        return

    for move in legal:
        nb = _apply_move(node.board, move, node.player)
        child = _TreeNode(nb, -node.player, move)
        node.children.append((move, -1, child))
        _expand_tree(child, depth - 1, max_leaves, leaf_list)


def _backpropagate(node: _TreeNode) -> float:
    """Negamax backpropagation through the expanded tree."""
    if node.is_leaf:
        return node.value
    best = -1e9
    for move, sign, child in node.children:
        v = sign * _backpropagate(child)
        if v > best: best = v
    node.value = best
    return best


def _batch_negamax(board_list, player, depth, model, device, max_leaves=50000):
    """
    GPU-friendly negamax: expand the entire tree in Python, then batch-evaluate
    all leaf nodes in a SINGLE model forward pass.

    Advantage over _negamax_ml: on GPU, one batch call of N leaves costs roughly
    the same as one single call — so N leaves are evaluated for the price of 1.
    On CPU the advantage is smaller but still 2-3x due to vectorisation.

    Args:
        max_leaves: cap on leaf count to avoid OOM (default 50k)
    """
    root = _TreeNode(board_list, player)
    leaf_list = []
    _expand_tree(root, depth, max_leaves, leaf_list)

    # ── Batch evaluate all leaves in one forward pass ────────────────────
    tensors = [_board_to_tensor(n.board, n.player) for n in leaf_list]
    batch = torch.from_numpy(np.stack(tensors)).to(device)
    with torch.no_grad():
        _, values = model(batch)      # (N,) value tensor
    values = values.cpu().numpy()

    for node, val in zip(leaf_list, values):
        node.value = float(val)

    _backpropagate(root)

    # Pick best move at root
    best_v, best_m = -1e9, None
    for move, sign, child in root.children:
        v = sign * (child.value if child.value is not None else 0.0)
        if v > best_v:
            best_v, best_m = v, move
    return best_v, best_m


# ── MLPAgent ─────────────────────────────────────────────────────────────
class MLPAgent(BaseAgent):
    """
    Othello agent powered by OthelloMLP (trained on Minimax Level-10 expert data).

    Search strategy (mode="negamax"):
        Uses the same Negamax + alpha-beta framework as the HC SearchAgent,
        but replaces the hand-crafted heuristic with the neural network's
        value_head at leaf nodes, and uses policy_head priors for move ordering.

        Time management: iterative deepening over odd depths (1→3→5→7→9).
        Only odd depths are searched to avoid the negamax horizon effect
        (even-depth searches can have systematic bias for the value head).
        The deepest fully-completed depth within the time budget is used.

    Modes:
        "policy"       — Single forward pass, policy head argmax. O(1) time.
                         Best for quick demonstrations or very fast games.
        "negamax"      — Negamax + alpha-beta + iterative deepening. Default.
                         Reaches depth 7 within a 3.0s budget on typical hardware.
        "batch_negamax" — Expand full tree in Python, evaluate ALL leaves in one
                         GPU batch forward pass. Ideal when GPU is available.
    """

    MAX_DEPTH = 9  # Max search depth (odd depths only: 1, 3, 5, 7, 9)

    def __init__(
        self,
        color: int,
        checkpoint_path: str,
        name: str | None = None,
        mode: Literal["policy", "negamax", "batch_negamax"] = "negamax",
        search_level: int = 5,
        device: str = "auto",
    ):
        if name is None:
            name = f"MLPAgent-{mode}"
        super().__init__(color=color, name=name)

        self.mode = mode

        # Map search_level (1–10) → actual search depth, always ODD.
        # Odd depths avoid negamax odd-even horizon oscillation.
        raw = max(1, min(search_level, self.MAX_DEPTH))
        self.depth = raw if raw % 2 == 1 else raw - 1

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(Path(checkpoint_path), map_location=self.device, weights_only=False)
        cfg  = ckpt.get("model_config", {})
        self.model = OthelloMLP(
            in_channels=cfg.get("in_channels", 4),
            hidden     =cfg.get("hidden",      [1024, 512, 512, 256, 256, 128]),
            dropout    =cfg.get("dropout",     0.1),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval().to(self.device)

        # TorchScript JIT compilation for 2-3x faster inference
        # (eliminates Python overhead in model.forward during tree search)
        try:
            dummy = torch.zeros(1, cfg.get("in_channels", 4), 8, 8).to(self.device)
            self.model = torch.jit.trace(self.model, dummy)
            self.model.eval()
        except Exception:
            pass  # Fallback to eager mode if trace fails

        val_acc = ckpt.get("best_val_acc", "?")
        epoch   = ckpt.get("epoch", "?")
        print(f"[MLPAgent] Loaded {checkpoint_path}  "
              f"mode={mode}  depth={self.depth}  val_acc={val_acc:.4f}  epoch={epoch}")

    def get_move(self, game_state, remain_time: float = 3.0):
        """
        Select best move within remain_time seconds.

        Args:
            game_state  : current GameState object
            remain_time : per-move time budget in seconds (default 3.0s, same as HC)

        Returns:
            (move, metrics) where move is (row, col) or None if no legal moves.
        """
        start    = time.perf_counter()
        deadline = start + max(0.1, remain_time - 0.05)  # 0.05s safety margin

        legal = game_state.get_legal_moves(self.color)
        if not legal:
            return None, self._metrics(start, 0, 0.0)

        board = game_state.board

        if self.mode == "policy":
            move, score = self._policy_move(board, legal)

        elif self.mode == "batch_negamax":
            score, move = _batch_negamax(board, self.color, self.depth,
                                         self.model, self.device)
            if move is None: move = legal[0]

        else:  # negamax — iterative deepening with time budget
            global _negamax_cache
            _negamax_cache = {}          # Reset transposition table each move

            best_move, best_score = legal[0], -1e9

            # Iterative deepening: search depth 1→3→5→7→9
            # Accept result only if the search fully completed before deadline.
            for depth in range(1, self.MAX_DEPTH + 1, 2):
                if time.perf_counter() >= deadline:
                    break

                score, move = _negamax_ml(
                    board, self.color, depth, -1e9, 1e9,
                    self.model, self.device, deadline
                )
                if move is not None and time.perf_counter() < deadline:
                    best_move, best_score = move, score

            move, score = best_move, best_score
            if move is None:
                move = legal[0]

        return move, self._metrics(start, len(legal), float(score))

    # ── Internal helpers ─────────────────────────────────────────────────

    def _policy_move(self, board, legal):
        """Select move by policy head argmax (no search)."""
        lmask = _legal_mask(board, self.color)
        s  = torch.from_numpy(_board_to_tensor(board, self.color)).unsqueeze(0).to(self.device)
        mk = torch.from_numpy(lmask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, v = self.model(s)
            p = p.masked_fill(mk <= 0, -1e9)
        idx  = int(p.argmax(1))
        move = (idx // 8, idx % 8)
        if move not in legal:
            move = random.choice(legal)
        return move, v.item()

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
