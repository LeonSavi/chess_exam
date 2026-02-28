import chess
import random
from scripts.player import TransformerPlayer

class RandomPlayer:
    """The 'Stupid AI' - plays completely random legal moves."""
    def get_move(self, board):
        return random.choice(list(board.legal_moves)).uci()

def play_match():
    # 1. Initialize our combatants
    print("🤖 Initializing the Chess God...")
    god_bot = TransformerPlayer(name="Chess-God", repo_id="LeoSavi/Chess-God-Transformer")
    
    stupid_bot = RandomPlayer()
    board = chess.Board()
    
    print("\n⚔️  MATCH START: Transformer vs. Random Bot ⚔️\n")
    print(board)
    
    # 2. Game Loop
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Transformer's turn
            move_uci = god_bot.get_move(board.fen())
            move_name = "Transformer"
        else:
            # Stupid AI's turn
            move_uci = stupid_bot.get_move(board)
            move_name = "Stupid AI"

        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                print(f"{move_name} plays: {move_uci}")
            else:
                print(f"⚠️ {move_name} attempted ILLEGAL move: {move_uci}. Skipping turn or forfeiting!")
                break
        except Exception as e:
            print(f"💥 Error processing move from {move_name}: {e}")
            break

    # 3. Results
    print("\n--- GAME OVER ---")
    print(f"Result: {board.result()}")
    if board.is_checkmate():
        winner = "Stupid AI" if board.turn == chess.WHITE else "Transformer"
        print(f"🏆 {winner} wins by CHECKMATE!")
    else:
        print("🤝 Draw or Forfeit.")
    
    print("\nFinal Board State:")
    print(board)

if __name__ == "__main__":
    play_match()