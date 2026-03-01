import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import optuna


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from datasets import load_dataset

from scripts.architecture import Transformer
from scripts.tokenizer import ChessTokenizer
from scripts.dataset import ChessDataset
from scripts.utils import YMLstudy

DATA_ROOT = 'data'
CSV_FILE = 'chess_moves'
N_TRIALS = 6
TRIAL_EPOCHS = 10

SEED = 1

YMLwriter = YMLstudy()


tokenizer = ChessTokenizer()

tactics_dataset = pd.DataFrame(load_dataset("ssingh22/chess-evaluations", "tactics"
                               )['train']).rename(columns={'FEN':'fen_before','Move':'move'})

hf_dataset = pd.DataFrame(load_dataset("bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
                                       )['train']).rename(columns={'FEN':'fen_before','Next move':'move'})

hf_data = pd.concat([tactics_dataset,hf_dataset],axis=0,ignore_index=True).reset_index(drop=True)

full_dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv",hf_data, tokenizer, sample_frac=0.01)# semple_frac it takes a sample

print(f'dataset sampled - n rows: {len(full_dataset)}')

def calculate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            src, tgt = batch["src"].to(device), batch["tgt"].to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(loader)


def objective(trial):
    # Suggestions (Keep d_model divisible by num_heads)
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [4, 8])
    num_layers = trial.suggest_int("num_layers", 3, 6)

    d_ff = d_model*4

    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    
    trial.set_user_attr("d_ff", d_ff)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(SEED))
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size)

    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff*4,
        max_seq_length=100,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(TRIAL_EPOCHS):
        model.train()
        for batch in train_loader:
            src, tgt = batch["src"].to(device), batch["tgt"].to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

    train_loss = calculate_loss(model, train_loader, criterion, device)
    val_loss = calculate_loss(model, val_loader, criterion, device)

    gap = abs(train_loss - val_loss)
    
    return val_loss, gap

if __name__ == "__main__":
    print("Starting Optuna Study...")
    study = optuna.create_study(
        directions=["minimize",'minimize'],
        storage="sqlite:///optuna-study/chess-transformer-study.db",
        load_if_exists=True,
        sampler = optuna.samplers.TPESampler(seed=SEED))
    
    study.optimize(objective, n_trials=N_TRIALS)
        
    YMLwriter.write_study(study)
    YMLwriter.write_best_param()
