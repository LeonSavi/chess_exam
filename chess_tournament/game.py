import chess
import random
from typing import Optional, Tuple, Dict

class Game:
    """
    Game orchestrates a match between two Player-like objects.
    Expectation: player.get_move(fen) -> str (UCI), "__NO_MOVES__", or None.
    Game is responsible for parsing/legality fallback counting.
    """

    def __init__(self, player_a, player_b, max_half_moves: int = 200):
        self.player_a = player_a
        self.player_b = player_b
        self.max_half_moves = max_half_moves

    def _apply_move_with_fallback(self, board: chess.Board, move_str: Optional[str]) -> Tuple[str, bool]:
        """
        Try to apply move_str (UCI) to the board.
        If move_str is None / invalid / illegal -> choose a random legal move and apply it.
        Returns:
            (applied_move_uci, was_fallback_used)
        Note: "__NO_MOVES__" sentinel should be handled by caller (play).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            # Terminal position — should be handled by caller before calling this
            raise RuntimeError("No legal moves available on board (terminal).")

        # None or empty -> fallback
        if not move_str:
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # If the player returned a tuple accidentally, take the first element
        if isinstance(move_str, tuple) and len(move_str) >= 1:
            move_str = move_str[0]

        # "__NO_MOVES__" sentinel — do not apply here (caller will react)
        if move_str == "__NO_MOVES__":
            return "__NO_MOVES__", False

        # Try parse UCI
        try:
            mv = chess.Move.from_uci(move_str)
        except Exception:
            # parsing failed -> fallback
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # If parsed move not legal in this position -> fallback
        if mv not in board.legal_moves:
            fallback = random.choice(legal_moves)
            board.push(fallback)
            return fallback.uci(), True

        # Legal move -> push
        board.push(mv)
        return mv.uci(), False

    def play(self, verbose: bool = False, force_colors: Optional[Tuple] = None
            ) -> Tuple[str, Dict[str, float], Dict[str, int]]:
        """
        Play a single game and return (result, scores, fallbacks)
        - result: "1-0", "0-1", or "1/2-1/2"
        - scores: {player_name: points}
        - fallbacks: {player_name: fallback_count} (only game-level fallbacks)
        """

        board = chess.Board()

        # Determine white/black
        if force_colors:
            white, black = force_colors
        else:
            players = [self.player_a, self.player_b]
            random.shuffle(players)
            white, black = players

        # initialize fallback counts (game-level)
        fallbacks = {white.name: 0, black.name: 0}

        if verbose:
            print(f"White: {white.name}  vs  Black: {black.name}")
            print(board, "\n")

        for ply in range(self.max_half_moves):
            if board.is_game_over():
                break

            current = white if board.turn == chess.WHITE else black
            fen = board.fen()

            # ask player for move
            try:
                mv_response = current.get_move(fen)
            except Exception as e:
                if verbose:
                    print(f"[{current.name}] get_move crashed: {e}")
                mv_response = None

            # normalize tuple return (legacy safety)
            if isinstance(mv_response, tuple) and len(mv_response) >= 1:
                proposed_move = mv_response[0]
            else:
                proposed_move = mv_response

            # special sentinel: true terminal (engine says no moves)
            if proposed_move == "__NO_MOVES__":
                winner = black if current == white else white
                if verbose:
                    print(f"{current.name} reported __NO_MOVES__ -> immediate loss for {current.name}")
                scores = {self.player_a.name: 0.0, self.player_b.name: 0.0}
                scores[winner.name] = 1.0
                result = "1-0" if winner == white else "0-1"
                return result, scores, fallbacks

            # apply move (Game-level fallback if needed)
            applied_move, parsing_fallback = self._apply_move_with_fallback(board, proposed_move)

            if parsing_fallback:
                fallbacks[current.name] += 1

            if verbose:
                print(f"PLY {ply:03d} | {current.name} plays {applied_move} {'(fallback)' if parsing_fallback else ''}")
                print(board, "\n")

        # game finished or max moves reached
        raw_result = board.result()  # possible values: "1-0", "0-1", "1/2-1/2", or "*"
        if raw_result == "*" or raw_result not in ["1-0", "0-1", "1/2-1/2"]:
            raw_result = "1/2-1/2"

        # map to scores for the two players (use their names)
        scores = {self.player_a.name: 0.0, self.player_b.name: 0.0}
        if raw_result == "1-0":
            # white (the variable) won
            scores[white.name] = 1.0
            scores[black.name] = 0.0
        elif raw_result == "0-1":
            scores[white.name] = 0.0
            scores[black.name] = 1.0
        else:
            scores[white.name] = 0.5
            scores[black.name] = 0.5

        if verbose:
            print("Game finished:", raw_result)
            print("Fallback counts:", fallbacks)

        return raw_result, scores, fallbacks
