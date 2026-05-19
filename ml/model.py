import torch
import torch.nn as nn

class EmotionRecognitionModel(nn.Module):
    def __init__(self, input_dim=180, num_classes=8):
        super(EmotionRecognitionModel, self).__init__()
        
        # 1D CNN for spatial feature extraction along the time axis
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.3)
        
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.3)
        
        # LSTM for temporal sequence modeling
        # Input to LSTM should be (batch, seq_len, features)
        # After conv1 and pool1 (len/2), conv2 and pool2 (len/4)
        # So seq_len is 100 / 4 = 25
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        
        # Fully connected layers for classification
        # Bidirectional LSTM means hidden size is 64 * 2 = 128
        self.fc1 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch, input_dim, max_len)
        
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Prepare for LSTM: swap dimensions to (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # We only need the output from the last time step
        last_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_out)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test model shape
    model = EmotionRecognitionModel()
    dummy_input = torch.randn(8, 180, 100) # batch_size=8, features=180, seq_len=100
    out = model(dummy_input)
    print(f"Model output shape: {out.shape}") # Should be (8, 8)
