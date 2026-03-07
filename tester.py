import chess
import random
from player import TransformerPlayer
import pandas as pd
import threading

from player import TransformerPlayer # Your custom player class
from chess_tournament.game import Game
from chess_tournament.players import EnginePlayer,RandomPlayer

from api import RAPID_API
import os

import concurrent.futures

NUM_MATCHES = 10

def run_single_match(match_id, player1, player2):
    """Function to run one game instance"""
    print(f"⚔️ [Match {match_id}] Starting: {player1.name} vs {player2.name}")
    
    # Create a fresh Game instance for each thread
    match = Game(player1, player2, max_half_moves=200)
    result = match.play(verbose=False)
    
    print(f"🏁 [Match {match_id}] Finished! Result: {result}")
    return result

if __name__ == "__main__":
    if "RAPIDAPI_KEY" not in os.environ:
        os.environ["RAPIDAPI_KEY"] = RAPID_API

    opponents = ['stockfish-weak', 'stockfish-mid', 'stockfish-strong', 'stockfish-gm']
    temperatures = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # Store results per temperature
    temp_results = {t: [] for t in temperatures}

    for t in temperatures:
        print(f"\n{'='*60}")
        print(f"🌡️  Testing Temperature: {t}")
        print(f"{'='*60}")
        
        for opponent in opponents:
            god_transformer = TransformerPlayer(name="TransformerGodPlayer", temperature=t)
            stockfish_gm = EnginePlayer(opponent, blunder_rate=0.0, ponder_rate=0.4)

            num_matches = NUM_MATCHES
            results = []

            print(f"🚀 Launching {num_matches} matches vs {opponent}...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_matches) as executor:
                futures = [executor.submit(run_single_match, i+1, god_transformer, stockfish_gm)
                          for i in range(num_matches)]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            wins = losses = draws = 0
            for res in results:
                score_str = float(res[1]['TransformerGodPlayer'])
                if score_str == 1.0:
                    wins += 1
                elif score_str == 0.0:
                    losses += 1
                else:
                    draws += 1

            win_rate = ((wins + 0.5 * draws) / len(results)) * 100
            temp_results[t].append({
                'opponent': opponent,
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'win_rate': win_rate
            })

    # ── FINAL SUMMARY TABLE PER TEMPERATURE ─────────────────────
    print("\n\n")
    print("=" * 75)
    print("        🏆 TEMPERATURE COMPARISON — TransformerGodPlayer")
    print("=" * 75)

    # Header
    print(f"{'Opponent':<20}", end="")
    for t in temperatures:
        print(f"  T={t:.2f}", end="")
    print()
    print("-" * 75)

    # Per opponent row
    for i, opponent in enumerate(opponents):
        print(f"{opponent:<20}", end="")
        for t in temperatures:
            wr = temp_results[t][i]['win_rate']
            print(f"  {wr:>5.1f}%", end="")
        print()

    print("-" * 75)

    # Overall row
    print(f"{'OVERALL':<20}", end="")
    best_temp = None
    best_wr = 0
    for t in temperatures:
        total_w = sum(r['wins'] for r in temp_results[t])
        total_d = sum(r['draws'] for r in temp_results[t])
        total_g = sum(len(opponents) * 20 for _ in [t])
        overall_wr = ((total_w + 0.5 * total_d) / (len(opponents) * 20)) * 100
        print(f"  {overall_wr:>5.1f}%", end="")
        if overall_wr > best_wr:
            best_wr = overall_wr
            best_temp = t
    print()
    print("=" * 75)
    print(f"\n🏅 BEST TEMPERATURE: {best_temp} (Overall WR: {best_wr:.1f}%)")
    print("=" * 75)


    # ── SAVE RESULTS TO CSV/EXCEL ────────────────────────────────
    rows = []
    for t in temperatures:
        for r in temp_results[t]:
            rows.append({
                'temperature': t,
                'opponent': r['opponent'],
                'wins': r['wins'],
                'losses': r['losses'],
                'draws': r['draws'],
                'win_rate': r['win_rate']
            })

        # Add overall row per temperature
        total_w = sum(r['wins'] for r in temp_results[t])
        total_l = sum(r['losses'] for r in temp_results[t])
        total_d = sum(r['draws'] for r in temp_results[t])
        total_g = len(opponents) * 20
        overall_wr = ((total_w + 0.5 * total_d) / total_g) * 100
        rows.append({
            'temperature': t,
            'opponent': 'OVERALL',
            'wins': total_w,
            'losses': total_l,
            'draws': total_d,
            'win_rate': overall_wr
        })

    df_results = pd.DataFrame(rows)

    # Save both formats
    df_results.to_csv('results/temperature_results.csv', index=False)
    print("\n💾 Results saved in results folder")
