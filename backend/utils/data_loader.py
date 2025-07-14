import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HealthDataset(Dataset):
    def __init__(self, csv_path, seq_len=10):
        df = pd.read_csv(csv_path)
        features = df[["heart_rate", "steps", "sleep", "stress_level", "spO2"]].values.astype(np.float32)
        labels = df["cardiovascular_risk"].values.astype(np.float32)
        # Normalize features (min-max)
        self.features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        y = self.labels[idx+self.seq_len-1]
        return torch.tensor(x), torch.tensor(y)

def get_dataloader(csv_path, batch_size=32, seq_len=10):
    dataset = HealthDataset(csv_path, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True) 