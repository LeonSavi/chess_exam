import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import optuna

import numpy as np
from sklearn.model_selection import KFold

from scripts.architecture import Transformer
from scripts.tokenizer import ChessTokenizer
from scripts.dataset import ChessDataset
from scripts.utils import YMLstudy

DATA_ROOT = 'data'
CSV_FILE = 'chess_moves_sample'
N_TRIALS = 50
TRIAL_EPOCHS = 5

SEED = 1

YMLwriter = YMLstudy()

def calculate_metrics(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            src, tgt = batch["src"].to(device), batch["tgt"].to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    return np.sqrt(avg_loss)


def objective(trial):
    d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    d_ff = trial.suggest_categorical("d_ff", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    tokenizer = ChessTokenizer()
    full_dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv", tokenizer)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    
    fold_rmses = []
    fold_gaps = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset)))):

        train_sub = Subset(full_dataset, train_idx)
        val_sub = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=batch_size)

        model = Transformer(
            src_vocab_size=tokenizer.vocab_size,
            tgt_vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
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

        train_rmse = calculate_metrics(model, train_loader, criterion, device)
        val_rmse = calculate_metrics(model, val_loader, criterion, device)

        fold_rmses.append(val_rmse)
        fold_gaps.append(abs(train_rmse - val_rmse))

    rmse_mean, rmse_std = np.mean(fold_rmses), np.std(fold_rmses)
    gap_mean, gap_std = np.mean(fold_gaps), np.std(fold_gaps)

    final_rmse_score = rmse_mean + 0.5 * rmse_std
    
    final_gap_score = gap_mean + 0.5 * gap_std

    return final_rmse_score, final_gap_score

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
