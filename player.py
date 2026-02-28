import os
import yaml
import torch
from huggingface_hub import hf_hub_download

from chess_tournament.players import Player

import sys
import os

# Get the directory where player.py is actually located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add that directory to the Python path so it can see the 'scripts' folder
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from scripts.tokenizer import ChessTokenizer
from scripts.architecture import Transformer


class TransformerPlayer(Player):
    def __init__(self,
                 name="TransformerGodPlayer",
                 repo_id="LeoSavi/Chess-God-Transformer"): 

        super().__init__(name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ChessTokenizer()

        weights_path = hf_hub_download(repo_id=repo_id, filename="TransformerGodPlayer.pth")
        configs = hf_hub_download(repo_id=repo_id, filename="opt-configs.yml")

        with open(configs, 'r') as file:
            settings = yaml.safe_load(file)

        self.model = Transformer(
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size,
            d_model=settings['d_model'],
            num_heads=settings['num_heads'],
            num_layers=settings['num_layers'],
            d_ff=settings['d_ff'],
            max_seq_length=100,
            dropout=settings['dropout']
        ).to(self.device)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval() 

    def get_move(self, fen: str) -> str:
        encoded_fen = self.tokenizer.encode(fen, is_target=False)
        src_tensor = torch.tensor(encoded_fen, dtype=torch.long).unsqueeze(0).to(self.device)
        
        target_tokens = [1]
        max_move_len = 7
        
        with torch.no_grad():
            for _ in range(max_move_len):
                tgt_tensor = torch.tensor(target_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                output = self.model(src_tensor, tgt_tensor)
                next_token_logits = output[0, -1, :] 
                next_token = next_token_logits.argmax().item()
                target_tokens.append(next_token)
                if next_token == 2:
                    break
                    
        predicted_move = self.tokenizer.decode(target_tokens)
        return predicted_move if predicted_move else "0000"