import os
import yaml
import torch
from huggingface_hub import hf_hub_download

from chess_tournament.players import Player

import sys
import os
import chess

import random

current_dir = os.path.dirname(os.path.abspath(__file__))

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

        config_filename = "opt-configs.yml"
        local_config = os.path.join(current_dir, config_filename)
        
        if os.path.exists(local_config):
            with open(local_config, 'r') as file:
                settings = yaml.safe_load(file)
        else:
            config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)
            with open(config_path, 'r') as file:
                settings = yaml.safe_load(file)

        weights_filename = "TransformerGodPlayer.pth"
        local_weights_option1 = os.path.join(current_dir, 'model', weights_filename)
        local_weights_option2 = os.path.join(current_dir, weights_filename)

        if os.path.exists(local_weights_option1):
            weights_path = local_weights_option1
        elif os.path.exists(local_weights_option2):
            weights_path = local_weights_option2
        else:
            weights_path = hf_hub_download(repo_id=repo_id, filename=f"{weights_filename}")

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
        board = chess.Board(fen)
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

        try:
            move = chess.Move.from_uci(predicted_move)
            if move in board.legal_moves:
                return move.uci()
        except:
            pass

        print(f"⚠️ {self.name} predicted illegal move {predicted_move}. Falling back to random.")
        return random.choice([m.uci() for m in board.legal_moves])
