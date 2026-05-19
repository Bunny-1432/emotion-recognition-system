import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from features import extract_features
from model import EmotionRecognitionModel
from dataset_prep import prepare_dataset

class EmotionDataset(Dataset):
    def __init__(self, csv_file, max_len=100):
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['emotion_code']
        
        # Extract features
        features = extract_features(file_path, max_len=self.max_len)
        
        # Fallback for failed feature extraction (e.g. corrupted file)
        if features is None:
            features = np.zeros((180, self.max_len))
            
        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor

def train_model(epochs=10, batch_size=32, lr=0.001):
    print("Preparing dataset...")
    # This will generate synthetic data if RAVDESS is not downloaded
    prepare_dataset()
    
    data_dir = Path(__file__).parent.parent / "data"
    
    train_dataset = EmotionDataset(data_dir / "train.csv")
    test_dataset = EmotionDataset(data_dir / "test.csv")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EmotionRecognitionModel(num_classes=8).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = Path(__file__).parent.parent / "best_model.pt"
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")

if __name__ == "__main__":
    train_model(epochs=5) # Reduced epochs for demonstration speed
