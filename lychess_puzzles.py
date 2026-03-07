import chess
import pandas as pd
from datasets import load_dataset

SEED = 1

def fix_puzzles(fen, moves_str):

    board = chess.Board(fen)
    moves = moves_str.strip().split()
    pairs = []

    for i, move_uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                break

            if i % 2 == 1: 
                pairs.append({
                    'fen_before': board.fen(),
                    'move': move_uci
                })

            board.push(move)

        except Exception:
            break

    return pairs


if '__main__'==__name__:

    puzzles = pd.DataFrame(load_dataset("lichess/chess-puzzles")['train'])
    
    puzzles_quality = puzzles[
        (puzzles['Popularity'] >= 90) &        
        (puzzles['NbPlays'] >= 3000) &        
        
        (puzzles['Rating'] >= 300) &
        (puzzles['Rating'] <= 2200) &
        
        (puzzles['Themes'].apply(lambda t: any(theme in (t if isinstance(t, list) 
            else t.split()) for theme in [
            'middlegame',     # most relevant to your tournament
            'crushing',       # blunder punishment
            'advantage',      # blunder punishment
            'mate',           # checkmates
            'mateIn1',        # checkmates
            'mateIn2',        # checkmates
            'fork',           # tactics
            'hangingPiece',   # blunder punishment
            'oneMove',        # simple decisive moves
        ])))    
    ]

    priority_themes = ['crushing', 'advantage', 'hangingPiece', 'fork']
    normal_themes = ['mate', 'mateIn1', 'mateIn2', 'middlegame', 'oneMove']

    priority = puzzles_quality[
        puzzles_quality['Themes'].apply(lambda t: any(
            theme in (t if isinstance(t, list) else t.split()) 
            for theme in priority_themes))
    ].sample(n=min(210000, len(puzzles_quality)), random_state=SEED)

    normal = puzzles_quality[
        puzzles_quality['Themes'].apply(lambda t: any(
            theme in (t if isinstance(t, list) else t.split()) 
            for theme in normal_themes))
    ].sample(n=min(210000, len(puzzles_quality)), random_state=SEED)

    puzzles_filtered = pd.concat([priority, normal]).drop_duplicates(subset=['PuzzleId']).reset_index(drop=True)

    print(f"Total puzzles: {len(puzzles):,}")

    # Unroll
    print("Unrolling puzzle sequences...")
    all_pairs = []
    for _, row in puzzles_filtered.iterrows():
        pairs = fix_puzzles(row['FEN'], row['Moves'])
        all_pairs.extend(pairs)

    df_training = pd.DataFrame(all_pairs)

    df_training['side'] = df_training['fen_before'].apply(lambda f: f.split()[1])
    side_balance = df_training['side'].value_counts()
    print(f"\nUnrolled pairs: {len(df_training):,}")
    print(f"White responses: {side_balance.get('w', 0):,} ({side_balance.get('w', 0)/len(df_training)*100:.1f}%)")
    print(f"Black responses: {side_balance.get('b', 0):,} ({side_balance.get('b', 0)/len(df_training)*100:.1f}%)")

    min_side = side_balance.min()
    df_balanced = pd.concat([
        df_training[df_training['side'] == 'w'].sample(n=min_side, random_state=SEED),
        df_training[df_training['side'] == 'b'].sample(n=min_side, random_state=SEED)
    ]).drop(columns='side').reset_index(drop=True)

    print(f"Balanced dataset: {len(df_balanced):,} pairs")

    df_balanced.to_csv('data/lichess_puzzles_unpacked.csv', index=False)
    print("Saved to data/lichess_puzzles_unpacked.csv")