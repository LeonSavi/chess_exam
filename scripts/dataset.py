import pandas as pd
import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self,
            csv_file,
            tokenizer,
            max_fen_len=90,
            max_move_len=7,
            players = ['Stockfish-GM','Stockfish-Strong']):

        self.data = pd.read_csv(csv_file)
        
        self.data = self.data[
             (self.data['player'].isin(players)) & 
            (self.data['fallback'] == False)
        ]
        
        self.tokenizer = tokenizer
        self.max_fen_len = max_fen_len
        self.max_move_len = max_move_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = str(row["fen_before"])
        move = str(row["move"]) 

        encoded_fen = self.tokenizer.encode(fen, is_target=False)
        encoded_move = self.tokenizer.encode(move, is_target=True)

        fen_padded = encoded_fen + [0] * (self.max_fen_len - len(encoded_fen))
        move_padded = encoded_move + [0] * (self.max_move_len - len(encoded_move))

        return {
            "src": torch.tensor(fen_padded, dtype=torch.long),
            "tgt": torch.tensor(move_padded, dtype=torch.long)
        }