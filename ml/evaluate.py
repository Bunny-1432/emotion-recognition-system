import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from model import EmotionRecognitionModel
from train import EmotionDataset
from dataset_prep import EMOTION_MAP

def evaluate_model():
    data_dir = Path(__file__).parent.parent / "data"
    test_csv = data_dir / "test.csv"
    
    if not test_csv.exists():
        print("Test data not found. Please run train.py first.")
        return
        
    test_dataset = EmotionDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = EmotionRecognitionModel(num_classes=8).to(device)
    model_path = Path(__file__).parent.parent / "best_model.pt"
    
    if not model_path.exists():
        print("Model file not found. Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Map index back to emotion string based on EMOTION_MAP
    # EMOTION_MAP values: 'neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'
    target_names = list(EMOTION_MAP.values())
    
    print("\nClassification Report:")
    report_text = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
    print(report_text)
    
    # Save metrics to JSON
    import json
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, output_dict=True)
    metrics_path = Path(__file__).parent.parent / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Emotion Recognition Confusion Matrix')
    cm_path = Path(__file__).parent.parent / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    evaluate_model()
