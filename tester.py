import chess
import random
from player import TransformerPlayer

from player import TransformerPlayer # Your custom player class
from chess_tournament.game import Game
from chess_tournament.players import EnginePlayer

from api import RAPID_API
import os

if __name__ == "__main__":

    if "RAPIDAPI_KEY" not in os.environ:
        os.environ["RAPIDAPI_KEY"] = RAPID_API

    # 1. Load the "Chess God"
    god_transformer = TransformerPlayer(name="Transformer-God")

    # 2. Setup the "Grandmaster" (Stockfish at full power)
    stockfish_gm = EnginePlayer("Stockfish-GM", blunder_rate=0.0, ponder_rate=0.0)

    # 3. Play the match
    # You can use the Game class from your library to manage the FENs and moves
    match = Game(god_transformer, stockfish_gm, max_half_moves=200)

    print(f"⚔️ Starting match: {god_transformer.name} (White) vs {stockfish_gm.name} (Black)")
    result = match.play(verbose=True) 

    print(f"🏁 Match Finished! Result: {result}")