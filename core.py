import copy
import numpy as np

class GameState:
    def __init__(self, board=None, current_turn=1):
        if board is None:
            self.board = [[0 for _ in range(8)] for _ in range(8)]
            self.board[3][3] = -1
            self.board[3][4] = 1
            self.board[4][4] = -1
            self.board[4][3] = 1
        else:
            self.board = copy.deepcopy(board)
        self.current_turn = current_turn

    def get_board_copy(self):
        """Returns a deep copy of the current GameState."""
        return GameState(self.board, self.current_turn)

    def calculate_score(self):
        """Returns a tuple (black_score, white_score). Black=1, White=-1."""
        black_score = sum(row.count(1) for row in self.board)
        white_score = sum(row.count(-1) for row in self.board)
        return black_score, white_score

    def get_legal_moves(self, color):
        """Returns a list of (row, col) for valid 'sandwich' moves."""
        valid_actions = []
        for i in range(8):
            for j in range(8):
                if self._is_valid_action(i, j, color):
                    valid_actions.append((i, j))
        return valid_actions

    def _is_valid_action(self, row, col, player):
        if self.board[row][col] != 0:
            return False
        
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                r = row + i
                c = col + j
                found_opponent = False
                while 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] == 0:
                        break
                    elif self.board[r][c] == player:
                        if found_opponent: return True
                        break
                    else:
                        found_opponent = True
                        r += i
                        c += j
        return False

    def apply_move(self, move, color):
        """Returns a new GameState with the move applied (Simulation support)."""
        new_state = self.get_board_copy()
        new_state.current_turn = -color
        if move is None:
            return new_state
            
        row, col = move[0], move[1]
        
        new_state.board[row][col] = color
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                r = row + i
                c = col + j
                flipped = False
                to_flip = []
                
                while 0 <= r < 8 and 0 <= c < 8:
                    if new_state.board[r][c] == 0:
                        break
                    elif new_state.board[r][c] == color:
                        flipped = True
                        break
                    else:
                        to_flip.append((r, c))
                        r += i
                        c += j
        
                if flipped:
                    for (r, c) in to_flip:
                        new_state.board[r][c] = color
                        
        return new_state

    def is_game_over(self):
        """Checks terminal conditions."""
        return len(self.get_legal_moves(1)) == 0 and len(self.get_legal_moves(-1)) == 0
