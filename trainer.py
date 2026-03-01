import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from scripts.architecture import Transformer
from scripts.tokenizer import ChessTokenizer
from scripts.dataset import ChessDataset

import pandas as pd
from datasets import load_dataset

from huggingface_hub import HfApi, login

from api import HF_TOKEN


REPO_ID = 'LeoSavi/Chess-God-Transformer'

DATA_ROOT = 'data'
MODEL_DIR = 'model'

NAME_MODEL = "TransformerGodPlayer.pth"

SAVE_PATH = os.path.join(MODEL_DIR, NAME_MODEL)
OPTUNA_PATH = "opt-configs.yml"

EPOCHS = 80

os.makedirs(MODEL_DIR, exist_ok=True)

# for data integration
tactics_dataset = pd.DataFrame(load_dataset("ssingh22/chess-evaluations", "tactics"
                               )['train']).rename(columns={'FEN':'fen_before','Move':'move'})

hf_dataset = pd.DataFrame(load_dataset("bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
                                       )['train']).rename(columns={'FEN':'fen_before','Next move':'move'})

hf_data = pd.concat([tactics_dataset,hf_dataset],axis=0,ignore_index=True).reset_index(drop=True)

def train_final_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(OPTUNA_PATH, 'r') as file:
        config = yaml.safe_load(file)

    tokenizer = ChessTokenizer()
    dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv", hf_data, tokenizer)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'])

    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_length=100,
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)
            
            # Using output.size(-1) to perfectly match dimensions
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        # Validation Check at the end of each epoch
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]
                
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Trained Model: {SAVE_PATH}")


def load_to_hf():

    login(token=HF_TOKEN)
    api = HfApi()

    api.upload_file(
        path_or_fileobj=SAVE_PATH,
        path_in_repo="TransformerGodPlayer.pth",
        repo_id=REPO_ID,
        commit_message="Update model"
    )

    api.upload_file(
        path_or_fileobj=OPTUNA_PATH,
        path_in_repo="opt-configs.yml",
        repo_id=REPO_ID,
        commit_message="Update Hypers"
    )

if __name__ == "__main__":
    train_final_model()
    load_to_hf()