import random as rd
import time

POSITION_WEIGHTS = [
    [1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00],
    [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
    [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
    [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
    [0.05, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.05],
    [0.10, 0.01, 0.05, 0.02, 0.02, 0.05, 0.01, 0.10],
    [-0.25, -0.25, 0.01, 0.01, 0.01, 0.01, -0.25, -0.25],
    [1.00, -0.25, 0.10, 0.05, 0.05, 0.10, -0.25, 1.00],
]

DIFFICULTY_TO_DEPTH = {
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 8,
}

DIFFICULTY_TO_RANDOMNESS = {
    1: 0.40,
    2: 0.25,
    3: 0.15,
    4: 0.08,
    5: 0.04,
    6: 0.02,
    7: 0.01,
    8: 0.00,
    9: 0.00,
    10: 0.00,
}

CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
CORNER_NEIGHBORS = {
    (0, 0): ((0, 1), (1, 0), (1, 1)),
    (0, 7): ((0, 6), (1, 7), (1, 6)),
    (7, 0): ((6, 0), (7, 1), (6, 1)),
    (7, 7): ((6, 7), (7, 6), (6, 6)),
}


def _board_key(board):
    return tuple(tuple(row) for row in board)


def _get_valid_moves_cached(board, turn, move_cache):
    key = (_board_key(board), turn)
    if key in move_cache:
        return move_cache[key]
    valid = get_valid_moves(board, turn)
    move_cache[key] = valid
    return valid


def _empty_count(board):
    count = 0
    for row in board:
        for cell in row:
            if cell == 0:
                count += 1
    return count


def _corner_score(board, root_player):
    my_corners = 0
    opp_corners = 0
    for r, c in CORNERS:
        if board[r][c] == root_player:
            my_corners += 1
        elif board[r][c] == -root_player:
            opp_corners += 1
    return my_corners - opp_corners


def _corner_closeness_penalty(board, root_player):
    penalty = 0
    for corner, neighbors in CORNER_NEIGHBORS.items():
        cr, cc = corner
        if board[cr][cc] != 0:
            continue
        for r, c in neighbors:
            if board[r][c] == root_player:
                penalty -= 1
            elif board[r][c] == -root_player:
                penalty += 1
    return penalty


def is_valid_move(board, row, col, turn):
    if board[row][col] != 0:
        return False
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            found_opponent = False
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] == 0:
                    break
                if board[r][c] == turn:
                    if found_opponent:
                        return True
                    break
                found_opponent = True
                r += i
                c += j
    return False


def get_valid_moves(board, turn):
    valid_moves = []
    for row in range(8):
        for col in range(8):
            if is_valid_move(board, row, col, turn):
                valid_moves.append((row, col))
    return valid_moves


def make_move(board, row, col, turn):
    new_board = [r[:] for r in board]
    new_board[row][col] = turn
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            r = row + i
            c = col + j
            flipped = False
            to_flip = []
            while 0 <= r < 8 and 0 <= c < 8:
                if new_board[r][c] == 0:
                    break
                if new_board[r][c] == turn:
                    flipped = True
                    break
                to_flip.append((r, c))
                r += i
                c += j
            if flipped:
                for fr, fc in to_flip:
                    new_board[fr][fc] = turn
    return new_board


def winner(board):
    total = sum(sum(row) for row in board)
    if total > 0:
        return 1
    if total < 0:
        return -1
    return 0


def heuristic(board, root_player, move_cache=None):
    if move_cache is None:
        move_cache = {}

    positional = 0.0
    my_count = 0
    opp_count = 0
    for row in range(8):
        for col in range(8):
            cell = board[row][col]
            if cell == root_player:
                positional += POSITION_WEIGHTS[row][col]
                my_count += 1
            elif cell == -root_player:
                positional -= POSITION_WEIGHTS[row][col]
                opp_count += 1

    mobility = len(_get_valid_moves_cached(board, root_player, move_cache)) - len(
        _get_valid_moves_cached(board, -root_player, move_cache)
    )
    disc_diff = my_count - opp_count

    corner_diff = _corner_score(board, root_player)
    corner_closeness = _corner_closeness_penalty(board, root_player)

    empties = _empty_count(board)
    if empties > 40:
        # Opening: mobility and corner control matter most.
        return float(
            positional * 14.0
            + mobility * 14.0
            + corner_diff * 80.0
            + corner_closeness * 22.0
            + disc_diff * 1.0
        )
    if empties > 14:
        # Midgame: balance positional quality and tactical mobility.
        return float(
            positional * 16.0
            + mobility * 12.0
            + corner_diff * 100.0
            + corner_closeness * 18.0
            + disc_diff * 2.0
        )

    # Endgame: disc advantage becomes decisive.
    return float(
        positional * 10.0
        + mobility * 6.0
        + corner_diff * 120.0
        + corner_closeness * 10.0
        + disc_diff * 12.0
    )


def _move_priority(board, move):
    r, c = move
    if (r, c) in CORNERS:
        return 10_000

    score = 0
    if r == 0 or r == 7 or c == 0 or c == 7:
        score += 300

    # Penalize dangerous X/C squares when corner is empty.
    if (r, c) in ((1, 1), (1, 6), (6, 1), (6, 6)):
        corner = (0 if r == 1 else 7, 0 if c == 1 else 7)
        if board[corner[0]][corner[1]] == 0:
            score -= 800
    if (r, c) in ((0, 1), (1, 0), (0, 6), (1, 7), (6, 0), (7, 1), (6, 7), (7, 6)):
        if r <= 1 and c <= 1:
            corner = (0, 0)
        elif r <= 1 and c >= 6:
            corner = (0, 7)
        elif r >= 6 and c <= 1:
            corner = (7, 0)
        else:
            corner = (7, 7)
        if board[corner[0]][corner[1]] == 0:
            score -= 450
    return score


def _order_moves(board, player_to_move, valid_moves, move_cache):
    scored = []
    for move in valid_moves:
        next_board = make_move(board, move[0], move[1], player_to_move)
        quick_priority = _move_priority(board, move)
        h = heuristic(next_board, player_to_move, move_cache)
        scored.append((quick_priority * 1000 + h, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [move for _, move in scored]


def _negamax(board, player_to_move, depth, alpha, beta, stats, deadline, move_cache):
    if time.perf_counter() >= deadline:
        stats["timed_out"] = True
        return heuristic(board, player_to_move, move_cache)

    stats["nodes_explored"] += 1

    valid_moves = _get_valid_moves_cached(board, player_to_move, move_cache)
    opp_valid_moves = _get_valid_moves_cached(board, -player_to_move, move_cache)

    if depth == 0 or (not valid_moves and not opp_valid_moves):
        if not valid_moves and not opp_valid_moves:
            game_winner = winner(board)
            if game_winner == player_to_move:
                return float("inf")
            if game_winner == -player_to_move:
                return float("-inf")
            return 0.0
        return heuristic(board, player_to_move, move_cache)

    if not valid_moves:
        return -_negamax(
            board,
            -player_to_move,
            depth - 1,
            -beta,
            -alpha,
            stats,
            deadline,
            move_cache,
        )

    best_value = float("-inf")
    ordered_moves = _order_moves(board, player_to_move, valid_moves, move_cache)

    for move in ordered_moves:
        child = make_move(board, move[0], move[1], player_to_move)
        value = -_negamax(
            child,
            -player_to_move,
            depth - 1,
            -beta,
            -alpha,
            stats,
            deadline,
            move_cache,
        )

        if value > best_value:
            best_value = value
        if value > alpha:
            alpha = value
        if alpha >= beta:
            stats["pruned_branches"] += 1
            break
        if stats["timed_out"]:
            break

    return best_value


def _iterative_search(board, player_to_move, max_depth, deadline):
    move_cache = {}
    valid_moves = _get_valid_moves_cached(board, player_to_move, move_cache)
    best_move = valid_moves[0]
    best_score = float("-inf")
    depth_reached = 0
    total_nodes = 0
    total_prunes = 0

    for depth in range(1, max_depth + 1):
        if time.perf_counter() >= deadline:
            break

        stats = {"nodes_explored": 0, "pruned_branches": 0, "timed_out": False}
        current_best_move = best_move
        current_best_score = float("-inf")

        ordered_moves = _order_moves(board, player_to_move, valid_moves, move_cache)
        for move in ordered_moves:
            if time.perf_counter() >= deadline:
                stats["timed_out"] = True
                break

            child = make_move(board, move[0], move[1], player_to_move)
            value = -_negamax(
                child,
                -player_to_move,
                depth - 1,
                float("-inf"),
                float("inf"),
                stats,
                deadline,
                move_cache,
            )

            if value > current_best_score:
                current_best_score = value
                current_best_move = move

            if stats["timed_out"]:
                break

        total_nodes += stats["nodes_explored"]
        total_prunes += stats["pruned_branches"]

        if not stats["timed_out"]:
            best_move = current_best_move
            best_score = current_best_score
            depth_reached = depth
        else:
            break

    if depth_reached == 0:
        best_score = heuristic(
            make_move(board, best_move[0], best_move[1], player_to_move),
            player_to_move,
            move_cache,
        )

    return best_move, {
        "nodes_explored": total_nodes,
        "pruned_branches": total_prunes,
        "depth_reached": depth_reached,
        "heuristic_score": best_score,
        "timed_out": time.perf_counter() >= deadline,
    }


def _compute_time_budget(remain_time, difficulty):
    requested = max(0.05, float(remain_time))

    # Keep a small safety margin to avoid crossing per-move hard limit due overhead.
    if requested <= 0.2:
        margin = 0.02
    elif requested <= 1.0:
        margin = 0.05
    elif requested <= 3.0:
        margin = 0.10
    else:
        margin = 0.15

    if difficulty >= 9:
        margin *= 0.6

    return max(0.05, requested - margin)


def select_move(cur_state, player_to_move, remain_time, difficulty=5):
    valid_moves = get_valid_moves(cur_state, player_to_move)
    if not valid_moves:
        return None, {
            "nodes_explored": 1,
            "pruned_branches": 0,
            "depth_reached": 0,
            "heuristic_score": 0.0,
            "timed_out": False,
            "difficulty": max(1, min(10, int(difficulty))),
        }

    difficulty = max(1, min(10, int(difficulty)))

    if DIFFICULTY_TO_RANDOMNESS[difficulty] > 0 and rd.random() < DIFFICULTY_TO_RANDOMNESS[difficulty]:
        move = rd.choice(valid_moves)
        return move, {
            "nodes_explored": 1,
            "pruned_branches": 0,
            "depth_reached": 0,
            "heuristic_score": heuristic(make_move(cur_state, move[0], move[1], player_to_move), player_to_move, {}),
            "timed_out": False,
            "difficulty": difficulty,
        }

    max_depth = DIFFICULTY_TO_DEPTH[difficulty]
    empties = _empty_count(cur_state)
    if difficulty >= 9 and empties <= 12:
        # High difficulty performs exact endgame search when possible.
        max_depth = max(max_depth, empties + 1)

    budget = _compute_time_budget(remain_time, difficulty)
    deadline = time.perf_counter() + budget

    best_move, metrics = _iterative_search(cur_state, player_to_move, max_depth, deadline)
    metrics["difficulty"] = difficulty
    return best_move, metrics

