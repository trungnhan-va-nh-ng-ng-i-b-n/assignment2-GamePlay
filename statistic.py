import time
from core import GameState
from Agents import RandomAgent

def simulate_game(agent1, agent2):
    """
    Run a headless game between two agents.
    Returns: 1 if agent1 wins, -1 if agent2 wins, 0 if draw.
    """
    game_state = GameState()
    
    while not game_state.is_game_over():
        turn = game_state.current_turn
        active_agent = agent1 if turn == 1 else agent2
        
        move, metrics = active_agent.get_move(game_state)
        
        # Apply move
        valid_moves = game_state.get_legal_moves(turn)
        if not valid_moves:
            valid_moves.append(None)
            
        if move not in valid_moves:
            print(f"{active_agent.name} attempted invalid move {move}. Defaulting to loss.")
            return -turn
            
        game_state = game_state.apply_move(move, turn)

    black_score, white_score = game_state.calculate_score()
    if black_score > white_score:
        return 1
    elif white_score > black_score:
        return -1
    else:
        return 0

def run_batch(agent1_class, agent2_class, num_games=10):
    print(f"Running {num_games} matches...")
    results = {"agent1_wins": 0, "agent2_wins": 0, "draws": 0}
    
    for i in range(num_games):
        # Alternate who goes first
        if i % 2 == 0:
            a1 = agent1_class(color=1, name="Agent1 (Black)")
            a2 = agent2_class(color=-1, name="Agent2 (White)")
            winner = simulate_game(a1, a2)
            if winner == 1:
                results["agent1_wins"] += 1
            elif winner == -1:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
        else:
            # agent1 plays White (-1), agent2 plays Black (1)
            a2 = agent2_class(color=1, name="Agent2 (Black)")
            a1 = agent1_class(color=-1, name="Agent1 (White)")
            winner = simulate_game(a2, a1)
            if winner == -1:
                results["agent1_wins"] += 1
            elif winner == 1:
                results["agent2_wins"] += 1
            else:
                results["draws"] += 1
                
        print(f"Game {i+1}/{num_games} finished.")
        
    print("\n--- RESULTS ---")
    print(f"Agent 1 Wins: {results['agent1_wins']}")
    print(f"Agent 2 Wins: {results['agent2_wins']}")
    print(f"Draws: {results['draws']}")

if __name__ == "__main__":
    run_batch(RandomAgent, RandomAgent, num_games=10)
