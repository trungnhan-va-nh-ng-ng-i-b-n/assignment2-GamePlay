import pygame as pg
import time
from concurrent.futures import ThreadPoolExecutor
from core import GameState
from Agents import SearchAgent, RandomAgent

BG_COLOR = (45, 55, 65) 
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
PANEL_COLOR = (38, 46, 56)
CARD_COLOR = (50, 60, 72)
CARD_BORDER_COLOR = (78, 92, 108)
BUTTON_COLOR = (76, 95, 112)
BUTTON_HOVER_COLOR = (95, 117, 138)
BUTTON_TEXT_COLOR = (240, 244, 248)
BUTTON_ACTIVE_COLOR = (70, 130, 108)
BUTTON_DISABLED_COLOR = (62, 70, 80)
BOARD_SIZE = 8
CELL_SIZE = 60
BOARD_PIXEL_SIZE = BOARD_SIZE * CELL_SIZE
DEFAULT_TOTAL_TIME = 180
MIN_TOTAL_TIME = 30
MAX_TOTAL_TIME = 600
TOTAL_TIME_STEP = 30
DEFAULT_SEARCH_TIME = 3.5
MIN_LEVEL = 1
MAX_LEVEL = 10
AUTO_SPEEDS = [
    ("Very Slow", 5.0),
    ("Slow", 3.5),
    ("Normal", 2.0),
    ("Fast", 1.0),
    ("Very Fast", 0.5),
]
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


def _build_search_agent(color, level, index, think_time=DEFAULT_SEARCH_TIME):
    return SearchAgent(
        color=color,
        difficulty=level,
        name=f"SearchAgent{index}-L{level}",
        remain_time=think_time,
    )


def _compute_agent_move(agent, game_state):
    return agent.get_move(game_state)


def _nearest_speed_index(think_time):
    return min(range(len(AUTO_SPEEDS)), key=lambda i: abs(AUTO_SPEEDS[i][1] - think_time))


def _draw_button(surface, rect, label, font, hovered, enabled=True, active=False):
    if not enabled:
        color = BUTTON_DISABLED_COLOR
    elif active:
        color = BUTTON_ACTIVE_COLOR
    else:
        color = BUTTON_HOVER_COLOR if hovered else BUTTON_COLOR
    pg.draw.rect(surface, color, rect, border_radius=8)
    text = font.render(label, True, BUTTON_TEXT_COLOR)
    text_rect = text.get_rect(center=rect.center)
    surface.blit(text, text_rect)


def _draw_preview_board(surface):
    board_rect = pg.Rect(0, 0, BOARD_PIXEL_SIZE, BOARD_PIXEL_SIZE)
    pg.draw.rect(surface, BG_COLOR, board_rect)

    for i in range(1, BOARD_SIZE):
        offset = i * CELL_SIZE
        pg.draw.line(surface, BLACK_COLOR, (offset, 0), (offset, BOARD_PIXEL_SIZE), 2)
        pg.draw.line(surface, BLACK_COLOR, (0, offset), (BOARD_PIXEL_SIZE, offset), 2)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell = INITIAL_STATE[row][col]
            if cell == -1:
                pg.draw.circle(surface, WHITE_COLOR, (col * CELL_SIZE + 30, row * CELL_SIZE + 30), 25)
            elif cell == 1:
                pg.draw.circle(surface, BLACK_COLOR, (col * CELL_SIZE + 30, row * CELL_SIZE + 30), 25)


def setup_match_menu(screen):
    clock = pg.time.Clock()
    panel_rect = pg.Rect(BOARD_PIXEL_SIZE, 0, WINDOW_WIDTH - BOARD_PIXEL_SIZE, WINDOW_HEIGHT)

    title_font = pg.font.Font(None, 52)
    label_font = pg.font.Font(None, 34)
    value_font = pg.font.Font(None, 44)
    info_font = pg.font.Font(None, 24)

    level1 = 5
    total_time = DEFAULT_TOTAL_TIME

    row1_controls_y = 150
    row2_controls_y = 280

    a1_minus = pg.Rect(610, row1_controls_y, 52, 52)
    a1_plus = pg.Rect(820, row1_controls_y, 52, 52)
    total_minus = pg.Rect(610, row2_controls_y, 52, 52)
    total_plus = pg.Rect(820, row2_controls_y, 52, 52)

    start_button = pg.Rect(610, 430, 262, 46)
    quit_button = pg.Rect(610, 484, 262, 38)

    while True:
        mouse_pos = pg.mouse.get_pos()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return None

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return None
                if event.key == pg.K_RETURN:
                    return level1, total_time
                if event.key == pg.K_q:
                    level1 = max(MIN_LEVEL, level1 - 1)
                if event.key == pg.K_w:
                    level1 = min(MAX_LEVEL, level1 + 1)
                if event.key == pg.K_z:
                    total_time = max(MIN_TOTAL_TIME, total_time - TOTAL_TIME_STEP)
                if event.key == pg.K_x:
                    total_time = min(MAX_TOTAL_TIME, total_time + TOTAL_TIME_STEP)

            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                if a1_minus.collidepoint(event.pos):
                    level1 = max(MIN_LEVEL, level1 - 1)
                elif a1_plus.collidepoint(event.pos):
                    level1 = min(MAX_LEVEL, level1 + 1)
                elif total_minus.collidepoint(event.pos):
                    total_time = max(MIN_TOTAL_TIME, total_time - TOTAL_TIME_STEP)
                elif total_plus.collidepoint(event.pos):
                    total_time = min(MAX_TOTAL_TIME, total_time + TOTAL_TIME_STEP)
                elif start_button.collidepoint(event.pos):
                    return level1, total_time
                elif quit_button.collidepoint(event.pos):
                    return None

        screen.fill(BG_COLOR)
        _draw_preview_board(screen)
        pg.draw.rect(screen, PANEL_COLOR, panel_rect)

        title = title_font.render("Match Setup", True, WHITE_COLOR)
        screen.blit(title, (610, 36))

        a1_label = label_font.render("Agent 1 Level", True, WHITE_COLOR)
        total_label = label_font.render("Total Time (s)", True, WHITE_COLOR)
        screen.blit(a1_label, (610, 106))
        screen.blit(total_label, (610, 236))

        _draw_button(screen, a1_minus, "-", value_font, a1_minus.collidepoint(mouse_pos))
        _draw_button(screen, a1_plus, "+", value_font, a1_plus.collidepoint(mouse_pos))
        _draw_button(screen, total_minus, "-", value_font, total_minus.collidepoint(mouse_pos))
        _draw_button(screen, total_plus, "+", value_font, total_plus.collidepoint(mouse_pos))

        level1_text = value_font.render(str(level1), True, WHITE_COLOR)
        total_text = value_font.render(str(total_time), True, WHITE_COLOR)

        level1_rect = level1_text.get_rect(center=(741, row1_controls_y + 26))
        total_rect = total_text.get_rect(center=(741, row2_controls_y + 26))
        screen.blit(level1_text, level1_rect)
        screen.blit(total_text, total_rect)

        _draw_button(screen, start_button, "Start Match", label_font, start_button.collidepoint(mouse_pos))
        _draw_button(screen, quit_button, "Exit", label_font, quit_button.collidepoint(mouse_pos))

        info = info_font.render("Q/W A1 | Z/X total | Enter", True, WHITE_COLOR)
        info_rect = info.get_rect(center=(BOARD_PIXEL_SIZE + (WINDOW_WIDTH - BOARD_PIXEL_SIZE) // 2, 566))
        screen.blit(info, info_rect)

        pg.display.flip()
        clock.tick(60)


def main():
    pg.init()
    screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pg.display.set_caption("Othello Agent Battle")

    while True:
        selected_levels = setup_match_menu(screen)
        if selected_levels is None:
            break

        level1, total_time = selected_levels

        agent1 = _build_search_agent(color=1, level=level1, index=1)
        agent2 = RandomAgent(color=-1, name="RandomAgent")

        player1 = Player(agent1, 1, total_time=total_time)
        player2 = Player(agent2, -1, total_time=total_time)

        game_state = GameState(INITIAL_STATE, 1)
        board = Board(game_state)

        game = Game(player1, player2, board, screen=screen)
        action = game.loop()
        if action != "restart":
            break

    pg.quit()

class Board:
    def __init__(self, game_state):
        self.game_state = game_state
        
    def draw_board(self, surface):
        surface.fill(BG_COLOR)
        for i in range(1, BOARD_SIZE):
            x = i * CELL_SIZE
            pg.draw.line(surface, BLACK_COLOR, (x, 0), (x, BOARD_PIXEL_SIZE), 2)
            pg.draw.line(surface, BLACK_COLOR, (0, x), (BOARD_PIXEL_SIZE, x), 2)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.game_state.board[row][col] == -1:
                    pg.draw.circle(surface, WHITE_COLOR, (col * CELL_SIZE + 30, row * CELL_SIZE + 30), 25)
                elif self.game_state.board[row][col] == 1:
                    pg.draw.circle(surface, BLACK_COLOR, (col * CELL_SIZE + 30, row * CELL_SIZE + 30), 25)
                    
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
    def __init__(self, agent, turn, total_time) -> None:
        self.agent = agent
        self.name = agent.name
        self.turn = turn
        self.time = float(total_time)
        self.last_move = None
        
    def move(self, game_state):
        ans, metrics = self.agent.get_move(game_state)
        elapsed_time = max(0.0, float(metrics.get('move_time', 0.0)))
        self.time -= elapsed_time
        if self.time < 0:
            return -1
        self.last_move = ans
        return ans

    def apply_move_result(self, move, metrics):
        elapsed_time = max(0.0, float(metrics.get('move_time', 0.0)))
        self.time -= elapsed_time
        if self.time < 0:
            return -1
        self.last_move = move
        return move

class Game:
    def __init__(self, player1, player2, board, screen=None):
        if not pg.get_init():
            pg.init()
        self.screen = screen if screen is not None else pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.player1 = player1
        self.player2 = player2
        self.board = board
        self.play_mode = "auto"
        self.speed_index = _nearest_speed_index(DEFAULT_SEARCH_TIME)
        self.pending_step = False
        self.game_over = False
        self.winner = 0
        self.end_message = ""
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_future = None
        self.pending_player = None
        self.pending_turn = None
        self.pending_started_at = 0.0

        self.title_font = pg.font.Font(None, 40)
        self.info_font = pg.font.Font(None, 31)
        self.control_title_font = pg.font.Font(None, 30)
        self.control_button_font = pg.font.Font(None, 30)
        self.control_info_font = pg.font.Font(None, 26)
        self.message_font = pg.font.Font(None, 28)

        panel_x = BOARD_PIXEL_SIZE + 28
        self.control_width = 400
        self.mode_button = pg.Rect(panel_x, 324, 190, 44)
        self.step_button = pg.Rect(panel_x + 210, 324, 190, 44)
        self.speed_minus_button = pg.Rect(panel_x, 414, 52, 44)
        self.speed_plus_button = pg.Rect(panel_x + 348, 414, 52, 44)
        self.new_match_button = pg.Rect(panel_x, 548, 190, 42)
        self.exit_button = pg.Rect(panel_x + 210, 548, 190, 42)
        self._apply_current_think_time()
        
    def end_check(self):
        return self.board.game_state.is_game_over()
        
    def win_check(self):
        black_score, white_score = self.board.game_state.calculate_score()
        if black_score > white_score: return 1
        if white_score > black_score: return -1
        return 0
        
    def draw_turn(self, player1, player2, playerturn):
        panel_rect = pg.Rect(BOARD_PIXEL_SIZE, 0, WINDOW_WIDTH - BOARD_PIXEL_SIZE, WINDOW_HEIGHT)
        pg.draw.rect(self.screen, PANEL_COLOR, panel_rect)

        card_x = BOARD_PIXEL_SIZE + 20
        card_w = WINDOW_WIDTH - card_x - 20
        card1 = pg.Rect(card_x, 20, card_w, 122)
        card2 = pg.Rect(card_x, 160, card_w, 122)

        pg.draw.rect(self.screen, CARD_COLOR, card1, border_radius=10)
        pg.draw.rect(self.screen, CARD_COLOR, card2, border_radius=10)
        pg.draw.rect(self.screen, CARD_BORDER_COLOR, card1, 2, border_radius=10)
        pg.draw.rect(self.screen, CARD_BORDER_COLOR, card2, 2, border_radius=10)

        p1_color = RED_COLOR if playerturn == 1 else WHITE_COLOR
        p2_color = RED_COLOR if playerturn == -1 else WHITE_COLOR

        p1_title = self.title_font.render(player1.name + ("'s Turn" if playerturn == 1 else ""), True, p1_color)
        p2_title = self.title_font.render(player2.name + ("'s Turn" if playerturn == -1 else ""), True, p2_color)

        p1_last_move = self.info_font.render("Last Move: " + str(player1.last_move), True, WHITE_COLOR)
        p2_last_move = self.info_font.render("Last Move: " + str(player2.last_move), True, WHITE_COLOR)
        p1_time = self.info_font.render(f"Time left: {player1.time:.5f}s", True, WHITE_COLOR)
        p2_time = self.info_font.render(f"Time left: {player2.time:.5f}s", True, WHITE_COLOR)

        self.screen.blit(p1_title, (card1.x + 16, card1.y + 10))
        self.screen.blit(p1_last_move, (card1.x + 16, card1.y + 54))
        self.screen.blit(p1_time, (card1.x + 16, card1.y + 84))

        self.screen.blit(p2_title, (card2.x + 16, card2.y + 10))
        self.screen.blit(p2_last_move, (card2.x + 16, card2.y + 54))
        self.screen.blit(p2_time, (card2.x + 16, card2.y + 84))
        
    def print_message(self, reason):
        short_reason = reason
        if len(short_reason) > 40:
            short_reason = short_reason[:37] + "..."
        text = self.message_font.render(short_reason, True, RED_COLOR)
        text_rect = text.get_rect()
        text_rect.center = (BOARD_PIXEL_SIZE + (WINDOW_WIDTH - BOARD_PIXEL_SIZE) // 2, 518)
        self.screen.blit(text, text_rect)

    def _shutdown_executor(self):
        if self.pending_future is not None and not self.pending_future.done():
            self.pending_future.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _clear_pending(self):
        self.pending_future = None
        self.pending_player = None
        self.pending_turn = None
        self.pending_started_at = 0.0

    def _start_move_request(self):
        if self.game_over or self.pending_future is not None:
            return
        turn = self.board.game_state.current_turn
        player = self.player1 if turn == 1 else self.player2
        state_copy = self.board.game_state.get_board_copy()

        self.pending_player = player
        self.pending_turn = turn
        self.pending_started_at = time.perf_counter()
        self.pending_future = self.executor.submit(_compute_agent_move, player.agent, state_copy)

    def _consume_ready_move(self):
        if self.pending_future is None or not self.pending_future.done():
            return

        future = self.pending_future
        player = self.pending_player
        turn = self.pending_turn
        self._clear_pending()

        try:
            move, metrics = future.result()
        except Exception:
            if turn in (1, -1):
                loser_name = player.name if player is not None else (self.player1.name if turn == 1 else self.player2.name)
                self._finalize_game(-turn, loser_name + " failed while computing move", loser_name + " failed while computing move")
            return

        if turn not in (1, -1) or player is None:
            return

        move = player.apply_move_result(move, metrics)
        if move == -1:
            self._finalize_game(-turn, player.name + " lose due to out of time", player.name + " lose due to out of time")
            return

        valid_move = self.board.update_board(move, turn)
        if not valid_move:
            self._finalize_game(-turn, player.name + " lose due to invalid move", player.name + " lose due to invalid move")
            return

        player_str = "Player 1" if turn == 1 else "Player 2"
        self._append_line('move.txt', player_str + " move: " + str(move))

        if self.end_check():
            winner = self.win_check()
            if winner == 1:
                self._finalize_game(1, "Player 1 wins!", "Player 1 wins!")
            elif winner == -1:
                self._finalize_game(-1, "Player 2 wins!", "Player 2 wins!")
            else:
                self._finalize_game(0, "Draw!", "Draw!")

    def _append_line(self, path, line):
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + "\n")

    def _toggle_mode(self):
        self.play_mode = "step" if self.play_mode == "auto" else "auto"
        self.pending_step = False

    def _apply_current_think_time(self):
        think_time = AUTO_SPEEDS[self.speed_index][1]
        self.player1.agent.remain_time = think_time
        self.player2.agent.remain_time = think_time

    def _change_speed(self, delta):
        new_index = max(0, min(len(AUTO_SPEEDS) - 1, self.speed_index + delta))
        if new_index != self.speed_index:
            self.speed_index = new_index
            self._apply_current_think_time()

    def _finalize_game(self, winner, message, output_log=None):
        self.game_over = True
        self.winner = winner
        self.end_message = message
        if output_log:
            self._append_line('Output.txt', output_log)

    def _draw_controls(self):
        mouse_pos = pg.mouse.get_pos()
        is_thinking = self.pending_future is not None

        mode_label = "Mode: AUTO" if self.play_mode == "auto" else "Mode: STEP"
        _draw_button(
            self.screen,
            self.mode_button,
            mode_label,
            self.control_button_font,
            self.mode_button.collidepoint(mouse_pos),
            enabled=not self.game_over,
            active=self.play_mode == "auto",
        )

        can_step = (self.play_mode == "step") and (not self.game_over) and (not is_thinking)
        _draw_button(
            self.screen,
            self.step_button,
            "Next Step",
            self.control_button_font,
            self.step_button.collidepoint(mouse_pos),
            enabled=can_step,
            active=self.play_mode == "step",
        )

        speed_name, think_time = AUTO_SPEEDS[self.speed_index]
        speed_label = self.control_title_font.render("Think Time", True, WHITE_COLOR)
        speed_value = self.control_info_font.render(f"{speed_name} ({think_time:.1f}s)", True, WHITE_COLOR)
        center_x = self.speed_minus_button.x + self.control_width // 2
        speed_label_rect = speed_label.get_rect(center=(center_x, 390))
        speed_value_rect = speed_value.get_rect(center=(center_x, 437))
        self.screen.blit(speed_label, speed_label_rect)
        self.screen.blit(speed_value, speed_value_rect)

        _draw_button(
            self.screen,
            self.speed_minus_button,
            "-",
            self.control_button_font,
            self.speed_minus_button.collidepoint(mouse_pos),
            enabled=not self.game_over,
            active=False,
        )
        _draw_button(
            self.screen,
            self.speed_plus_button,
            "+",
            self.control_button_font,
            self.speed_plus_button.collidepoint(mouse_pos),
            enabled=not self.game_over,
            active=False,
        )

        if self.game_over:
            over = self.control_title_font.render("GAME OVER", True, RED_COLOR)
            over_rect = over.get_rect(center=(self.speed_minus_button.x + self.control_width // 2, 476))
            self.screen.blit(over, over_rect)

            _draw_button(
                self.screen,
                self.new_match_button,
                "New Match",
                self.control_button_font,
                self.new_match_button.collidepoint(mouse_pos),
                enabled=True,
                active=False,
            )
            _draw_button(
                self.screen,
                self.exit_button,
                "Exit",
                self.control_button_font,
                self.exit_button.collidepoint(mouse_pos),
                enabled=True,
                active=False,
            )
        else:
            hint_text = "Space: next step | Tab: mode | +/-: think time"
            hint = self.control_info_font.render(hint_text, True, WHITE_COLOR)
            self.screen.blit(hint, (BOARD_PIXEL_SIZE + 34, 476))

            if is_thinking:
                elapsed = time.perf_counter() - self.pending_started_at
                thinking = self.control_info_font.render(f"Thinking... {elapsed:.2f}s", True, WHITE_COLOR)
                self.screen.blit(thinking, (BOARD_PIXEL_SIZE + 34, 504))

    def _handle_key(self, event):
        if event.key == pg.K_ESCAPE:
            return "quit"

        if self.game_over:
            if event.key == pg.K_r:
                return "restart"
            return None

        if event.key == pg.K_TAB:
            self._toggle_mode()
        elif event.key == pg.K_SPACE and self.play_mode == "step":
            self.pending_step = True
        elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
            self._change_speed(-1)
        elif event.key in (pg.K_EQUALS, pg.K_KP_PLUS, pg.K_PLUS):
            self._change_speed(1)
        return None

    def _handle_click(self, pos):
        if self.game_over:
            if self.new_match_button.collidepoint(pos):
                return "restart"
            if self.exit_button.collidepoint(pos):
                return "quit"
            return None

        if self.mode_button.collidepoint(pos):
            self._toggle_mode()
        elif self.step_button.collidepoint(pos) and self.play_mode == "step" and self.pending_future is None:
            self.pending_step = True
        elif self.speed_minus_button.collidepoint(pos):
            self._change_speed(-1)
        elif self.speed_plus_button.collidepoint(pos):
            self._change_speed(1)
        return None
        
    def loop(self):
        clock = pg.time.Clock()
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self._shutdown_executor()
                    return "quit"
                if event.type == pg.KEYDOWN:
                    action = self._handle_key(event)
                    if action in ("restart", "quit"):
                        self._shutdown_executor()
                        return action
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._handle_click(event.pos)
                    if action in ("restart", "quit"):
                        self._shutdown_executor()
                        return action

            if not self.game_over:
                self._consume_ready_move()

                if self.play_mode == "step":
                    if self.pending_step and self.pending_future is None:
                        self.pending_step = False
                        self._start_move_request()
                else:
                    if self.pending_future is None:
                        self._start_move_request()

            self.board.draw_board(self.screen)
            self.draw_turn(self.player1, self.player2, self.board.game_state.current_turn)
            self._draw_controls()

            if self.game_over and self.end_message:
                self.print_message(self.end_message)

            pg.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    main()
