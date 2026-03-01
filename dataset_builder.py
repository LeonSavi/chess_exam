import os
import time
import random
import threading
import numpy as np
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from chess_tournament.game import Game
from chess_tournament.players import EnginePlayer, RandomPlayer


from api import RAPID_API

ROOT = 'data'
max_concurrent_games = 20
num_games_total = 10000-235-154-192-544-343

os.makedirs(ROOT, exist_ok=True)

csv_filepath = os.path.join(ROOT, "chess_moves.csv")
lock = threading.Lock() 

if "RAPIDAPI_KEY" not in os.environ:
    os.environ["RAPIDAPI_KEY"] = RAPID_API

good_players = [
    EnginePlayer("Stockfish-GM", blunder_rate=0.0, ponder_rate=0.0),
    EnginePlayer("Stockfish-Strong", blunder_rate=0.0, ponder_rate=0.1)
    ]

pool_of_players = [
    EnginePlayer("Stockfish-GM", blunder_rate=0.0, ponder_rate=0.1),
    EnginePlayer("Stockfish-Strong", blunder_rate=0.0, ponder_rate=0.15),

    EnginePlayer("Stockfish-GM", blunder_rate=0.01, ponder_rate=0.15),
    EnginePlayer("Stockfish-Strong", blunder_rate=0.05, ponder_rate=0.2),

    EnginePlayer("Stockfish-Mid", blunder_rate=0.1, ponder_rate=0.3),
    EnginePlayer("Stockfish-Weak", blunder_rate=0.15, ponder_rate=0.4),

    EnginePlayer("Stockfish-Mid", blunder_rate=0.15, ponder_rate=0.4),
    EnginePlayer("Stockfish-Weak", blunder_rate=0.3, ponder_rate=0.7),

    RandomPlayer("Chaos-Bot")
    ]


games_done = 0

def play_single_game(game_id, num_games_total):
    print(f"Starting Game {game_id}...")

    global games_done

    main_player = random.choice(good_players)
    
    opponent = random.choice(pool_of_players)

    if random.choice([True,False]): # sometimes the GM is black other times is white
        w,b = main_player,opponent
    else:
        w,b = opponent,main_player
        
    game = Game(w, b, max_half_moves=700)

    game.play(
        verbose=False, 
        force_colors=(w, b), 
        log_moves=False, 
        log_to_file=csv_filepath
    )

    with lock:
        games_done += 1
        print(f"Finished Game {game_id} ({w.name} vs {b.name}),  ")
        print(f"Progress: {games_done}/{num_games_total}")


if '__main__'==__name__:

    with ThreadPoolExecutor(max_workers=max_concurrent_games) as executor:
        for i in range(1, num_games_total + 1):
            executor.submit(play_single_game, i, num_games_total)
            time.sleep(3) 
