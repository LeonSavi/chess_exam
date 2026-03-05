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


REPO_ID = 'LeoSavi/Chess-God-Transformer'

DATA_ROOT = 'data'
MODEL_DIR = 'model'

SAVE_BASE_PATH = os.path.join(MODEL_DIR, 'TransformerGodPlayerBase.pth')

SAVE_PATH = os.path.join(MODEL_DIR, "TransformerGodPlayer.pth")
OPTUNA_PATH = "opt-configs.yml"

EPOCHS = 25
SEED = 1

SHARE = 1.0

os.makedirs(MODEL_DIR, exist_ok=True)

# for data integration
print('Downloading Data')
pre_tactics_dataset = pd.DataFrame(load_dataset("ssingh22/chess-evaluations", "tactics"
                               )['train'])

pre_tactics_dataset['eval_num'] = pre_tactics_dataset['Evaluation'].apply(parse_eval)

white = pre_tactics_dataset[(pre_tactics_dataset['eval_num']>0) & (pre_tactics_dataset['eval_num']<2000)].copy()
black = pre_tactics_dataset[(pre_tactics_dataset['eval_num']<0) & (pre_tactics_dataset['eval_num']>-2000)].copy()

white_check = pre_tactics_dataset[(pre_tactics_dataset['eval_num']>=2000)].copy()
black_check = pre_tactics_dataset[(pre_tactics_dataset['eval_num']<=-2000)].copy()

target = int(min(len(white)*SHARE,len(black)*SHARE))

white,black = white.sample(n=target,random_state=SEED),black.sample(n=target,random_state=SEED)

tactics_dataset =  pd.concat([white,black],ignore_index=True
                             ).reset_index(drop=True).dropna(subset=['FEN','Move']
                             ).rename(columns={'FEN':'fen_before','Move':'move'})

hf_dataset = pd.DataFrame(load_dataset("bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
                                       )['train']).rename(columns={'FEN':'fen_before','Next move':'move'})

hf_data = pd.concat([tactics_dataset,hf_dataset],axis=0,ignore_index=True).reset_index(drop=True)

# for fine-tuning: keeping a small portion of old data to avoid forgetting

small_white,small_black,small_hf = (
    white.sample(frac=0.01,random_state=SEED),
    black.sample(frac=0.01,random_state=SEED),
    hf_dataset.sample(frac=0.5,random_state=SEED) 
)
check_dataset =  pd.concat([white_check,black_check,small_white,small_black],ignore_index=True).reset_index(drop=True).rename(
    columns={'FEN':'fen_before','Move':'move'})

check_dataset = pd.concat([check_dataset,small_hf],ignore_index=True).reset_index(drop=True)

def train_base_model():
    print('--- Starting Base Model Training (with AMP Speed Boost) ---')
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

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        losses['EPOCH'].append(epoch+1)
        losses['training_loss'].append(avg_train_loss)
        losses['validation_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), SAVE_BASE_PATH)
    pd.DataFrame(losses).to_csv('training_losses_base.csv')
    print(f"Trained Model: {SAVE_BASE_PATH}")


def fine_tune_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    losses = {'EPOCH': [], 'training_loss':[],'validation_loss':[]}

    with open(OPTUNA_PATH, 'r') as file:
        config = yaml.safe_load(file)

    tokenizer = ChessTokenizer()
    dataset = ChessDataset(f"{DATA_ROOT}/chess_moves.csv",hf_data=check_dataset, tokenizer=tokenizer,sample_frac=0.2)
    
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

    model.load_state_dict(torch.load(SAVE_BASE_PATH, map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    fine_tune_lr = config['lr'] * 0.1
    optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
    
    scaler = GradScaler()
    acc_steps = 4 
    FT_EPOCHS = 4 
    
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
        
        print(f"Fine-Tune Epoch {epoch+1}/{FT_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
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

if __name__ == "__main__":
    print('base model training')
    train_base_model()
    print('fine tuning')
    fine_tune_model()
    load_to_hf()