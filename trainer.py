import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from scripts.architecture import Transformer
from scripts.tokenizer import ChessTokenizer
from scripts.dataset import ChessDataset
from scripts.utils import parse_eval

from torch.amp import autocast, GradScaler

import pandas as pd
from datasets import load_dataset

from huggingface_hub import HfApi, login

from api import HF_TOKEN

from datetime import datetime


REPO_ID = 'LeoSavi/Chess-God-Transformer'

DATA_ROOT = 'data'
MODEL_DIR = 'model'

SAVE_BASE_PATH = os.path.join(MODEL_DIR, 'TransformerGodPlayerBase.pth')

SAVE_PATH = os.path.join(MODEL_DIR, "TransformerGodPlayer.pth")
OPTUNA_PATH = "opt-configs.yml"

EPOCHS = 35
SEED = 1

SHARE = 1.0

LOAD_DATA = False

col_2_keep = ['fen_before','move']

os.makedirs(MODEL_DIR, exist_ok=True)

# for data integration
if LOAD_DATA:
    print('Downloading Data')
    pre_tactics_dataset = pd.DataFrame(load_dataset("ssingh22/chess-evaluations", "tactics"
                                )['train']).rename(columns={'FEN':'fen_before','Move':'move'})

    pre_tactics_dataset['eval_num'] = pre_tactics_dataset['Evaluation'].apply(parse_eval)

    white = pre_tactics_dataset[(pre_tactics_dataset['eval_num']>0) & (pre_tactics_dataset['eval_num']<2000)].copy()
    black = pre_tactics_dataset[(pre_tactics_dataset['eval_num']<0) & (pre_tactics_dataset['eval_num']>-2000)].copy()

    white_check = pre_tactics_dataset[(pre_tactics_dataset['eval_num']>=2000)].copy()
    black_check = pre_tactics_dataset[(pre_tactics_dataset['eval_num']<=-2000)].copy()

    target = int(min(len(white)*SHARE,len(black)*SHARE))

    white,black = white.sample(n=target,random_state=SEED),black.sample(n=target,random_state=SEED)

    tactics_dataset =  pd.concat([white[col_2_keep],black[col_2_keep]],ignore_index=True
                                ).reset_index(drop=True).dropna(
                                    subset=['fen_before','move']
                                )

    hf_dataset = pd.DataFrame(load_dataset("bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
                                        )['train']).rename(columns={'FEN':'fen_before','Next move':'move'})

    hf_data = pd.concat([tactics_dataset,hf_dataset[col_2_keep]],axis=0,ignore_index=True).reset_index(drop=True)

    # for fine-tuning: keeping a small portion of old data to avoid forgetting

    check_dataset =  pd.concat([white_check[col_2_keep],black_check[col_2_keep]],
                            ignore_index=True).reset_index(drop=True).rename(
                                columns={'FEN':'fen_before','Move':'move'})

    lychess_dataset = pd.read_csv(
                            'data/lichess_puzzles_unpacked.csv'
                            )

    check_dataset = pd.concat([lychess_dataset,check_dataset],
                            ignore_index=True).drop_duplicates().reset_index(drop=True)

    buffer_target = int(len(check_dataset) * 0.18 / (1 - 0.18)) 
    print(f'buffer target: {buffer_target}')
    finetuning_len_tactics = int((buffer_target - len(hf_dataset)) / 2)

    finetuning_df = pd.concat([
        check_dataset,
        hf_dataset,
        white.sample(n=finetuning_len_tactics,random_state=SEED),
        black.sample(n=finetuning_len_tactics,random_state=SEED),
        ]
        ,ignore_index=True).sample( # quick shuffle
            frac=1, random_state=SEED).reset_index(drop=True) 

    print(f'finetuning-data: {len(finetuning_df)}')



def train_base_model():
    print(f'--- Starting Base Model Training (time: {datetime.now().strftime("%H:%M:%S")})---')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    losses = {'EPOCH': [], 'training_loss':[],'validation_loss':[]}

    with open(OPTUNA_PATH, 'r') as file:
        config = yaml.safe_load(file)

    tokenizer = ChessTokenizer()
    dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv", hf_data, tokenizer)

    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    print(f'training on {train_size} data')


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
    
    scaler = GradScaler('cuda')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS,
                eta_min=0.00005)
            
    best_val_loss = float('inf') # set this high
    patience = 0
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
            
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                tgt_input = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]
                
                with autocast(device_type='cuda'):
                    output = model(src, tgt_input)
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
                
                total_val_loss += loss.item()
        
        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        losses['EPOCH'].append(epoch+1)
        losses['training_loss'].append(avg_train_loss)
        losses['validation_loss'].append(avg_val_loss)
        
        print(f"(time: {datetime.now().strftime("%H:%M:%S")}) Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 0.002:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), SAVE_BASE_PATH)  # save best achieved model
        else:
            patience += 1
            if patience >= 3:
                print('Early stopping — no improvement for 3 epochs')
                break

    # torch.save(model.state_dict(), SAVE_BASE_PATH)
    pd.DataFrame(losses).to_csv('training_losses_base.csv')
    print(f"Trained Model: {SAVE_BASE_PATH}")


def fine_tune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    losses = {'EPOCH': [], 'training_loss':[],'validation_loss':[]}

    with open(OPTUNA_PATH, 'r') as file:
        config = yaml.safe_load(file)

    tokenizer = ChessTokenizer()
    dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv",hf_data=finetuning_df, tokenizer=tokenizer,sample_frac=0.20)
    
    train_size = int(0.9 * len(dataset))
    print(f'training on {train_size}')
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

    model.load_state_dict(torch.load(SAVE_BASE_PATH, map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    fine_tune_lr = config['lr'] * 0.1
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    
    scaler = GradScaler('cuda')
    acc_steps = 4 

    FT_EPOCHS = 3
    patience = 0
    best_val_loss=float('inf')

    # Freeze first 3 layers of the encoder and decoder
    for i, layer in enumerate(model.encoder_layers):
        if i < 3:
            for param in layer.parameters():
                param.requires_grad = False
    for i, layer in enumerate(model.decoder_layers):
        if i < 3:
            for param in layer.parameters():
                param.requires_grad = False
    
    for epoch in range(FT_EPOCHS):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]

            with autocast(device_type='cuda'):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_expected.reshape(-1))
                loss_to_save = loss.item() 
                loss = loss / acc_steps
            
            scaler.scale(loss).backward()

            if (i + 1) % acc_steps == 0:

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            total_train_loss += loss_to_save

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            with autocast(device_type='cuda'): 
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

        losses['EPOCH'].append(epoch+1)
        losses['training_loss'].append(avg_train_loss)
        losses['validation_loss'].append(avg_val_loss)
        
        print(f"(time: {datetime.now().strftime("%H:%M:%S")}) Fine-Tune Epoch {epoch+1}/{FT_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 0.003:
            best_val_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), SAVE_PATH)  # save best achieved model
        else:
            patience += 1
            if patience >= 2:
                print('Early stopping — no improvement for 2 epochs')
                break

    pd.DataFrame(losses).to_csv('training_losses_finetuned.csv')
    print(f"Fine-Tuned Model Saved: {SAVE_PATH}")



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

    api.upload_file(
        path_or_fileobj="README.md",      # path to your local README
        path_in_repo="README.md",          # where it goes on HuggingFace
        repo_id=REPO_ID,
        commit_message="Update README"
    )

    api.upload_file(
        path_or_fileobj="charts/training_curves.png",
        path_in_repo="charts/training_curves.png",
        repo_id=REPO_ID,
        commit_message="Add training curves chart"
    )

if __name__ == "__main__":
    print('base model training')
    # train_base_model()
    print('fine tuning')
    # fine_tune_model()
    load_to_hf()