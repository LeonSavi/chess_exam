import time
import random
from typing import List, Dict, Any
from .game import Game
from .players import EnginePlayer
import gc

def instantiate_participant(desc):
    raise NotImplementedError("Provide instantiate_participant()")

def destroy_instance(inst):
    raise NotImplementedError("Provide destroy_instance()")

def round_robin_tournament(
    players: List,
    games_per_pair: int = 2,
    verbose: bool = False,
    engine_break: float = 3.0,
    engine_break_jitter: float = 1.0,
    max_half_moves: int = 150
) -> Dict:
    """
    Round-robin: every unordered pair plays `games_per_pair` games.
    Returns a summary dict with scores, games_played, fallbacks, leaderboard.
    """

    names = [p.name for p in players]
    scores = {n: 0.0 for n in names}
    fallbacks = {n: 0 for n in names}
    games_played = {n: 0 for n in names}

    n = len(players)
    pairs = []
    # build unordered pairs (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((players[i], players[j]))

    total_matches = len(pairs) * games_per_pair
    print(f"🏁 Round-robin: {len(players)} players, {len(pairs)} pairs, {games_per_pair} games/pair -> {total_matches} matches\n")

    match_idx = 0
    for p1, p2 in pairs:
        for g in range(games_per_pair):
            match_idx += 1
            # alternate colors: even g -> p1 white, odd g -> p2 white
            if g % 2 == 0:
                white, black = p1, p2
            else:
                white, black = p2, p1

            #if verbose:
            print(f"> Match {match_idx}/{total_matches}: {white.name} (white) vs {black.name} (black) ... ", end="", flush=True)

            game = Game(p1, p2, max_half_moves=max_half_moves)
            result, match_scores, match_fallbacks = game.play(verbose=verbose, force_colors=(white, black))

            # update stats
            scores[p1.name] += match_scores[p1.name]
            scores[p2.name] += match_scores[p2.name]
            fallbacks[p1.name] += match_fallbacks[p1.name]
            fallbacks[p2.name] += match_fallbacks[p2.name]
            games_played[p1.name] += 1
            games_played[p2.name] += 1

            #if verbose:
            print(f"Result: {result} | Scores: {match_scores} | Fallbacks: {match_fallbacks}")

            # if engine involved, pause a bit
            if isinstance(p1, EnginePlayer) or isinstance(p2, EnginePlayer):
                pause = engine_break + random.uniform(0.0, engine_break_jitter)
                if verbose:
                    print(f"[pause] Waiting {pause:.2f}s before next match")
                time.sleep(pause)

    # leaderboard sort by points, then fewer fallbacks
    def sort_key(nm):
        return (-scores[nm], fallbacks[nm], random.random())

    leaderboard = sorted(names, key=sort_key)

    print("\n🏆 FINAL ROUND-ROBIN LEADERBOARD 🏆")
    print("Rank | Name | Points | Games | Fallbacks")
    for rank, name in enumerate(leaderboard, start=1):
        print(f"{rank:>2} | {name:<15} | {scores[name]:>5.1f} | {games_played[name]:>5} | {fallbacks[name]:>8}")

    return {
        "scores": scores,
        "games_played": games_played,
        "fallbacks": fallbacks,
        "leaderboard": leaderboard
    }

def swiss_tournament(
    participant_descs: List[Dict[str,Any]],
    instantiate_fn,
    destroy_fn,
    n_rounds: int = 3,
    games_per_pairing: int = 2,
    max_half_moves: int = 150,
    engine_break: float = 0.0,
):
    """
    Swiss tournament using PER-MATCH instantiation.

    participant_descs : lightweight descriptors (students + baselines)
    instantiate_fn    : function(desc) -> Player instance
    destroy_fn        : function(instance)
    """

    names = [p["name"] for p in participant_descs]

    scores = {n: 0.0 for n in names}
    fallbacks = {n: 0 for n in names}
    opponents = {n: [] for n in names}
    past_pairs = set()

    print(f"🏁 Swiss tournament ({len(names)} players, {n_rounds} rounds)")

    for rnd in range(1, n_rounds + 1):
        print(f"\n=== Round {rnd} ===")

        # Sort by score
        sorted_names = sorted(names, key=lambda n: (-scores[n], random.random()))

        used = set()
        round_pairings = []

        for i, p1 in enumerate(sorted_names):
            if p1 in used:
                continue

            opponent_found = None
            for p2 in sorted_names[i+1:]:
                if p2 in used:
                    continue
                if frozenset({p1,p2}) not in past_pairs:
                    opponent_found = p2
                    break

            if opponent_found is None:
                for p2 in sorted_names[i+1:]:
                    if p2 not in used:
                        opponent_found = p2
                        break

            if opponent_found:
                round_pairings.append((p1, opponent_found))
                used.add(p1)
                used.add(opponent_found)
                past_pairs.add(frozenset({p1, opponent_found}))

        print("Pairings:", round_pairings)

        # ---- PLAY MATCHES ----
        for p1_name, p2_name in round_pairings:

            desc1 = next(d for d in participant_descs if d["name"] == p1_name)
            desc2 = next(d for d in participant_descs if d["name"] == p2_name)

            for game_idx in range(games_per_pairing):

                print(f"> {p1_name} vs {p2_name} (game {game_idx+1}) ... ", end="")

                p1 = instantiate_fn(desc1)
                p2 = instantiate_fn(desc2)

                try:
                    game = Game(p1, p2, max_half_moves=max_half_moves)
                    result, match_scores, match_fallbacks = game.play(verbose=False)

                finally:
                    destroy_fn(p1)
                    destroy_fn(p2)

                scores[p1_name] += match_scores[p1_name]
                scores[p2_name] += match_scores[p2_name]
                fallbacks[p1_name] += match_fallbacks[p1_name]
                fallbacks[p2_name] += match_fallbacks[p2_name]

                opponents[p1_name].append(p2_name)
                opponents[p2_name].append(p1_name)

                print(f"{result}")

                if engine_break > 0:
                    time.sleep(engine_break)

    # ---- FINAL SORT ----
    leaderboard = sorted(
        names,
        key=lambda n: (-scores[n], fallbacks[n], random.random())
    )

    print("\n🏆 FINAL LEADERBOARD 🏆")
    for rank, name in enumerate(leaderboard, start=1):
        print(f"{rank:>2}. {name:<20}  {scores[name]:>5.1f} pts  | fallbacks {fallbacks[name]}")

    return {
        "scores": scores,
        "fallbacks": fallbacks,
        "leaderboard": leaderboard
    }
    
def run_tournament(player_a, player_b, n_games=4, verbose=False, max_half_moves=200):
    results = {
        player_a.name: {"points": 0.0, "wins": 0, "draws": 0, "fallbacks": 0},
        player_b.name: {"points": 0.0, "wins": 0, "draws": 0, "fallbacks": 0},
    }

    print(f"🏁 Tournament: {player_a.name} vs {player_b.name}")
    print(f"Games: {n_games}\n")

    for game_idx in range(1, n_games + 1):
        print(f"--- Game {game_idx} ---")

        game = Game(player_a, player_b, max_half_moves)
        result, scores, fallbacks = game.play(verbose=verbose)

        # Aggregate stats
        for player_name in results.keys():
            results[player_name]["points"] += scores[player_name]
            results[player_name]["fallbacks"] += fallbacks[player_name]

        if result == "1-0":
            winner = max(scores, key=scores.get)
            results[winner]["wins"] += 1

        elif result == "0-1":
            winner = max(scores, key=scores.get)
            results[winner]["wins"] += 1

        else:
            results[player_a.name]["draws"] += 1
            results[player_b.name]["draws"] += 1

        print("Result:", result)
        print("Scores:", scores)
        print("Fallbacks:", fallbacks, "\n")

    # Final summary
    print("\n🏆 FINAL SUMMARY 🏆")

    for player_name, stats in results.items():
        print(f"\n{player_name}")
        print(f"Points: {stats['points']:.1f}")
        print(f"Wins: {stats['wins']}")
        print(f"Draws: {stats['draws']}")
        print(f"Fallbacks used: {stats['fallbacks']}")
