# backend/model.py
"""
Stub for LSTM/Temporal CNN model for time-series health risk prediction.
"""
import torch
import torch.nn as nn
import pandas as pd

class HealthRiskLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Placeholder for Temporal CNN, if needed 

sleep = pd.read_csv("fitabaseexampledata/SleepClassic/ID 1003_minuteSleep_20171001_20171007.csv")
print(sleep.columns)
print(sleep.head()) 