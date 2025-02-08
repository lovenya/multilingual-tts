import torch
import torch.nn as nn

class DurationPredictor(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, dropout=0.1):
        super(DurationPredictor, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (B, T, hidden_size) → transpose to (B, hidden_size, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        # Project back: transpose to (B, T, hidden_size) then linear layer.
        x = self.linear(x.transpose(1, 2))  # (B, T, 1)
        return x.squeeze(-1)

class Predictor(nn.Module):
    """
    Generic predictor module used for both pitch and energy.
    """
    def __init__(self, hidden_size, kernel_size=3, dropout=0.1):
        super(Predictor, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (B, T, hidden_size) → transpose to (B, hidden_size, T)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        x = self.linear(x.transpose(1, 2))
        return x.squeeze(-1)

# Aliases for clarity:
PitchPredictor = Predictor
EnergyPredictor = Predictor
