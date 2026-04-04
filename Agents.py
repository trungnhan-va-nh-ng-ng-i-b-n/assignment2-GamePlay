import random as rd
import time

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
