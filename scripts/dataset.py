import pandas as pd
from datasets import load_dataset

import torch
from torch.utils.data import Dataset

class ChessDataset(Dataset):
    def __init__(self,
            csv_file = None,
            hf_data = None,
            tokenizer = None,
            max_fen_len=90,
            max_move_len=7,
            sample_frac:float = None):
        

        self.ref_col = ['fen_before','move']
        
        if csv_file:
            self.data = pd.read_csv(csv_file)

            players = ['Stockfish-GM','Stockfish-Strong']

            self.data = self.data[
                (self.data['player'].isin(players)) & 
                (self.data['fallback'] == False)
            ]

            if sample_frac:
                self.data = self.data.sample(frac=sample_frac,random_state=1)
        
        else:
            self.data =  pd.DataFrame(columns=self.ref_col)

        self._add_hf_data(hf_data)
        
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
    
    def _add_hf_data(self,hf_data):

        if len(self.data)>0:
            to_concat = [self.data[self.ref_col],hf_data[self.ref_col]]

            self.data = pd.concat(
                to_concat,
                axis=0,
                ignore_index=True
                ).drop_duplicates(
                ).reset_index(drop=True)
        else:
            self.data = hf_data[self.ref_col
                                ].drop_duplicates().reset_index(drop=True)


