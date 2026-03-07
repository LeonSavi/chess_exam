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
                 name:str,
                 model_id:str="TransformerGodPlayer",
                 repo_id:str="LeoSavi/Chess-God-Transformer",
                 temperature:float = 0.8): 

        super().__init__(name)

        self.temperature = temperature
        
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

        actual_model_name = "TransformerGodPlayer.pth" 
        weights_filename = f"{model_id}.pth"

        local_weights_option1 = os.path.join(current_dir, 'model', actual_model_name)
        local_weights_option2 = os.path.join(current_dir, actual_model_name)

        if os.path.exists(local_weights_option1):
            weights_path = local_weights_option1
        elif os.path.exists(local_weights_option2):
            weights_path = local_weights_option2
        else:
            try:
                # Try to download the file named after the player
                weights_path = hf_hub_download(repo_id=repo_id, filename=weights_filename)
            except Exception:
                print('downloading from HF')
                weights_path = hf_hub_download(repo_id=repo_id, filename=actual_model_name)
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


    def generate_move(self,src_tensor,temp_ttp):
        target_tokens = [1]  
        with torch.no_grad():
            for _ in range(7): #max move len
                tgt_tensor = torch.tensor(target_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                output = self.model(src_tensor, tgt_tensor)
                next_token_logits = output[0, -1, :]
                probs = torch.softmax(next_token_logits / temp_ttp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                if next_token == 2: 
                    break
                target_tokens.append(next_token)
        return self.tokenizer.decode(target_tokens)
    
    def score_legal_moves(self,src_tensor,legal_moves):
        """
        Score each legal move by summing log probs of its characters.

        """
        move_scores = {}

        with torch.no_grad():
            for move_uci in legal_moves:
                move_tokens = [self.tokenizer.char_to_int[c] 
                               for c in move_uci 
                               if c in self.tokenizer.char_to_int]

                log_prob_sum = 0.0
                target_tokens = [1]

                for expected_token in move_tokens:
                    tgt_tensor = torch.tensor(
                        target_tokens, dtype=torch.long
                    ).unsqueeze(0).to(self.device)

                    output = self.model(src_tensor, tgt_tensor)
                    log_probs = torch.log_softmax(
                        output[0, -1, :] / self.temperature, dim=-1
                    )
                    log_prob_sum += log_probs[expected_token].item()
                    target_tokens.append(expected_token)

                move_scores[move_uci] = log_prob_sum

        return move_scores


    def get_move(self, fen: str) -> str:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        encoded_fen = self.tokenizer.encode(fen, is_target=False)
        src_tensor = torch.tensor(encoded_fen, dtype=torch.long).unsqueeze(0).to(self.device)

        # rise temperature and try to get an answer if illegal
        for n in range(4):
            temp = min(self.temperature + n*0.05, 0.85) # warm up
            predicted_move = self.generate_move(src_tensor,temp)
            try:
                move = chess.Move.from_uci(predicted_move)
                if move in board.legal_moves:
                    return move.uci()
            except:
                pass

        # check probability for each legal move
        try:
            scores = self.score_legal_moves(src_tensor,legal_moves)
            best_move = max(scores, key=scores.get)
            return best_move
        except:
            pass

        return random.choice(legal_moves)