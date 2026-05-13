import random as rd
import time
from algorithms.Minimax import select_move

class BaseAgent:
    def __init__(self, color, name="SearchAgent"):
        self.color = color
        self.name = name

    def get_move(self, game_state):
        
        raise NotImplementedError("Subclasses must implement get_move()")

class RandomAgent(BaseAgent):
    def __init__(self, color, name="RandomAgent"):
        super().__init__(color, name)

    def get_move(self, game_state):
        start_time = time.perf_counter()
        valid_moves = game_state.get_legal_moves(self.color)
        
        move = None
        if valid_moves:
            move = rd.choice(valid_moves)
            
        move_time = time.perf_counter() - start_time
        metrics = {
            "move_time": move_time,
            "nodes_explored": 1,
            "heuristic_score": 0
        }
        return move, metrics


class SearchAgent(BaseAgent):
    def __init__(self, color, difficulty=5, name=None, remain_time=3.0):
        if name is None:
            name = f"SearchAgent-L{difficulty}"
        super().__init__(color, name)
        self.difficulty = max(1, min(10, int(difficulty)))
        self.remain_time = float(remain_time)

    def get_move(self, game_state):
        start_time = time.perf_counter()

        move, search_metrics = select_move(
            game_state.board,
            self.color,
            self.remain_time,
            difficulty=self.difficulty,
        )

        move_time = time.perf_counter() - start_time
        metrics = {
            "move_time": move_time,
            "nodes_explored": search_metrics.get("nodes_explored", 0),
            "heuristic_score": search_metrics.get("heuristic_score", 0.0),
            "depth_reached": search_metrics.get("depth_reached", 0),
            "pruned_branches": search_metrics.get("pruned_branches", 0),
            "difficulty": search_metrics.get("difficulty", self.difficulty),
        }
        return move, metrics
