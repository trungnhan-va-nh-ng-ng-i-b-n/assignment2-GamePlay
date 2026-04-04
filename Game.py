import pygame as pg
import time
from core import GameState
from Agents import RandomAgent

BG_COLOR = (45, 55, 65) 
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
BOARD_SIZE = 8
DELAY_TIME = 2
TOTAL_TIME = 60
LIMIT_TIME = 3
INITIAL_STATE = [[0,0,0,0,0,0,0,0], 
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,-1,1,0,0,0],
                 [0,0,0,1,-1,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0]]
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600

def main():
    agent1 = RandomAgent(color=1, name="RandomAgent1")
    agent2 = RandomAgent(color=-1, name="RandomAgent2")
    
    player1 = Player(agent1, 1)
    player2 = Player(agent2, -1)

    game_state = GameState(INITIAL_STATE, 1)
    board = Board(game_state)
    
    game = Game(player1, player2, board)
    game.loop()

class Board:
    def __init__(self, game_state):
        self.game_state = game_state
        
    def draw_board(self, surface):
        surface.fill(BG_COLOR)
        for i in range(1, BOARD_SIZE):
            x = i * 60
            pg.draw.line(surface, BLACK_COLOR, (x, 0), (x, 480), 2)
            pg.draw.line(surface, BLACK_COLOR, (0, x), (480, x), 2)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.game_state.board[row][col] == -1:
                    pg.draw.circle(surface, WHITE_COLOR, (col * 60 + 30, row * 60 + 30), 25)
                elif self.game_state.board[row][col] == 1:
                    pg.draw.circle(surface, BLACK_COLOR, (col * 60 + 30, row * 60 + 30), 25)
                    
    def update_board(self, position, player):
        valid_moves = self.game_state.get_legal_moves(player)
        if not valid_moves:
            valid_moves.append(None)
            
        if position not in valid_moves: 
            return False
            
        self.game_state = self.game_state.apply_move(position, player)
        return True
        
    def get_board(self):
        return self.game_state.board

class Player:
    def __init__(self, agent, turn) -> None:
        self.agent = agent
        self.name = agent.name
        self.turn = turn
        self.time = TOTAL_TIME
        self.last_move = None
        
    def move(self, game_state):
        ans, metrics = self.agent.get_move(game_state)
        elapsed_time = metrics.get('move_time', 0.0)
        
        if elapsed_time > LIMIT_TIME:
            return -1
        self.time -= elapsed_time
        if self.time < 0:
            return -1
        self.last_move = ans
        return ans

class Game:
    def __init__(self, player1, player2, board):
        pg.init()
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.player1 = player1
        self.player2 = player2
        self.board = board
        
    def end_check(self):
        return self.board.game_state.is_game_over()
        
    def win_check(self):
        black_score, white_score = self.board.game_state.calculate_score()
        if black_score > white_score: return 1
        if white_score > black_score: return -1
        return 0
        
    def draw_turn(self, player1, player2, playerturn):
        font = pg.font.Font(None, 30)
        player1_name = font.render(player1.name, True, WHITE_COLOR)
        player2_name = font.render(player2.name, True, WHITE_COLOR)
        player1_last_move = font.render("Last Move: "+str(player1.last_move), True, WHITE_COLOR)
        player2_last_move = font.render("Last Move: "+str(player2.last_move), True, WHITE_COLOR)
        player1_time = font.render(f"Time left: {player1.time:.5f}s", True, WHITE_COLOR)
        player2_time = font.render(f"Time left: {player2.time:.5f}s", True, WHITE_COLOR)
        
        player1_name_rect = player1_name.get_rect()
        player2_name_rect = player2_name.get_rect()
        player1_last_move_rect = player1_last_move.get_rect()
        player2_last_move_rect = player2_last_move.get_rect()
        player1_time_rect = player1_time.get_rect()
        player2_time_rect = player2_time.get_rect()
        
        player1_name_rect.center = (WINDOW_WIDTH - 200, 50)
        player2_name_rect.center = (WINDOW_WIDTH - 200, WINDOW_HEIGHT//2)
        player1_last_move_rect.center = (WINDOW_WIDTH - 100, 100)
        player2_last_move_rect.center = (WINDOW_WIDTH - 100, WINDOW_HEIGHT//2+50)
        player1_time_rect.center = (WINDOW_WIDTH - 100, 150)
        player2_time_rect.center = (WINDOW_WIDTH - 100, WINDOW_HEIGHT//2+100)
        
        if playerturn == 1:
            player1_name = font.render(player1.name+"'s Turn", True, RED_COLOR)
        else:
            player2_name = font.render(player2.name+"'s Turn", True, RED_COLOR)
            
        self.screen.blit(player1_name, player1_name_rect)
        self.screen.blit(player2_name, player2_name_rect)
        self.screen.blit(player1_last_move, player1_last_move_rect)
        self.screen.blit(player2_last_move, player2_last_move_rect)
        self.screen.blit(player1_time, player1_time_rect)
        self.screen.blit(player2_time, player2_time_rect)
        
    def print_message(self, reason):
        font = pg.font.Font(None, 30)
        text = font.render(reason, True, RED_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (WINDOW_WIDTH-300, WINDOW_HEIGHT-100)
        self.screen.blit(text, text_rect)
        
    def loop(self):
        self.board.draw_board(self.screen)
        self.draw_turn(self.player1, self.player2, self.board.game_state.current_turn)
        pg.display.flip()
        looping = False
        winner = 0
        
        while not looping:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    looping = False
                    
            turn = self.board.game_state.current_turn
            
            if turn == 1:
                move = self.player1.move(self.board.game_state)
                if move == -1:
                    winner = -1
                    self.print_message(self.player1.name+" lose due to out of time")
                    with open('Output.txt', 'a', encoding='utf-8') as f:
                        f.write(self.player1.name+" lose due to out of time\n")
                    pg.display.flip()
                    time.sleep(DELAY_TIME)
                    break
            else: 
                move = self.player2.move(self.board.game_state)
                if move == -1:
                    winner = 1
                    self.print_message(self.player2.name+" lose due to out of time")
                    with open('Output.txt', 'a', encoding='utf-8') as f:
                        f.write(self.player2.name+" lose due to out of time\n")
                    pg.display.flip()
                    time.sleep(DELAY_TIME)
                    break
                    
            valid_move = self.board.update_board(move, turn)
            if not valid_move:
                winner = -turn
                name_lose = self.player1.name if turn == 1 else self.player2.name
                self.print_message(name_lose + " lose due to invalid move")
                with open('Output.txt', 'a', encoding='utf-8') as f:
                    f.write(name_lose + " lose due to invalid move\n")
                pg.display.flip()
                time.sleep(DELAY_TIME)
                break

            self.board.draw_board(self.screen)
            self.draw_turn(self.player1, self.player2, self.board.game_state.current_turn)
            pg.display.flip()
            # time.sleep(1) # Removed sleep to let bot vs bot run faster
            
            player_str = "Player 1" if turn == 1 else "Player 2"
            with open('move.txt', 'a', encoding='utf-8') as f:
                f.write(player_str + " move: " + str(move) + '\n')
                
            looping = self.end_check()
            if looping:
                winner = self.win_check()
                if winner == 1:
                    self.print_message("Player 1 wins!")
                elif winner == -1:
                    self.print_message("Player 2 wins!")
                else:
                    self.print_message("Draw!")
                pg.display.flip()
                time.sleep(DELAY_TIME)
            else:
                pass

if __name__ == "__main__":
    main()
