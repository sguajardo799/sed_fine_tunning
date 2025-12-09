import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for wavs, labels, _ in pbar:
        wavs = wavs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(wavs) # [Batch, Time, Classes]
        
        # Resize labels to match logits time dimension if necessary
        # The dataset generates labels based on 'time_resolution', which we hope matches the model.
        # If not, we might need to interpolate.
        if logits.shape[1] != labels.shape[1]:
            # Simple interpolation for now if mismatch
            # labels: [Batch, Time_Target, Classes] -> [Batch, Classes, Time_Target]
            labels_t = labels.permute(0, 2, 1)
            labels_t = torch.nn.functional.interpolate(labels_t, size=logits.shape[1], mode='nearest')
            labels = labels_t.permute(0, 2, 1)
            
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for wavs, labels, _ in pbar:
            wavs = wavs.to(device)
            labels = labels.to(device)
            
            logits = model(wavs)
            
            if logits.shape[1] != labels.shape[1]:
                labels_t = labels.permute(0, 2, 1)
                labels_t = torch.nn.functional.interpolate(labels_t, size=logits.shape[1], mode='nearest')
                labels = labels_t.permute(0, 2, 1)
            
            loss = criterion(logits, labels)
            running_loss += loss.item()
            
            # Binarize predictions for F1
            preds = torch.sigmoid(logits) > 0.5
            
            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_targets.append(labels.cpu().numpy().reshape(-1))
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return running_loss / len(loader), f1

def train(model, train_loader, val_loader, epochs, lr, device, save_dir):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_f1 = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print("Saved best model.")
            
        # Save checkpoint
        torch.save(model.state_dict(), save_dir / "last_model.pth")
