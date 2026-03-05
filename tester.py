import chess
import random
from player import TransformerPlayer

import threading

from player import TransformerPlayer # Your custom player class
from chess_tournament.game import Game
from chess_tournament.players import EnginePlayer,RandomPlayer

from api import RAPID_API
import os

import concurrent.futures

def run_single_match(match_id, player1, player2):
    """Function to run one game instance"""
    print(f"⚔️ [Match {match_id}] Starting: {player1.name} vs {player2.name}")
    
    # Create a fresh Game instance for each thread
    match = Game(player1, player2, max_half_moves=500)
    result = match.play(verbose=False)
    
    print(f"🏁 [Match {match_id}] Finished! Result: {result}")
    return result

if __name__ == "__main__":
    if "RAPIDAPI_KEY" not in os.environ:
        os.environ["RAPIDAPI_KEY"] = RAPID_API

    opponents = ['stockfish-weak', 'stockfish-mid', 'stockfish-strong', 'stockfish-gm']
    for opponent in opponents:
        # 1. Initialize Players
        god_transformer = TransformerPlayer(name="TransformerGodPlayer", temperature=0.8)
        stockfish_gm = EnginePlayer(opponent, blunder_rate=0.0, ponder_rate=0.1)

        num_matches = 100
        results = []

        # 2. Use ThreadPoolExecutor to run matches in parallel
        print(f"🚀 Launching {num_matches} matches in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_matches) as executor:
            # Create a list of 'future' tasks
            futures = [executor.submit(run_single_match, i+1, god_transformer, stockfish_gm) 
                    for i in range(num_matches)]
            
            # Collect results as they finish
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        # 2. Process results
        total_games = len(results)
        wins = 0
        losses = 0
        draws = 0
        
        for res in results:
            score_str = float(res[1]['TransformerGodPlayer'])
            if score_str == 1.0:
                wins += 1
            elif score_str == 0.0:
                losses += 1
            else:
                draws += 1

        # Standard Chess Win Rate: (Wins + 0.5 * Draws) / Total
        total_points = wins + (0.5 * draws)
        win_rate = (total_points / total_games) * 100

        # 3. Final Summary
        print(f"\n📊 --- TOURNAMENT SUMMARY: {opponent} ---")
        print(f"Total Games: {total_games}")
        print("-" * 30)
        print(f"✅ Wins:   {wins}")
        print(f"❌ Losses: {losses}")
        print(f"🤝 Draws:  {draws}")
        print("-" * 30)
        print(f"📈 TOTAL WIN RATE: {win_rate:.2f}%")
        print("-" * 30)