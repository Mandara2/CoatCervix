# data/dataloaders.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import settings

class NPYDataset(Dataset):
    """
    dataset personalized for numpy arrays, converting them to PyTorch tensors and rearranging dimensions as needed.
    """
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.transpose(X, (0, 3, 1, 2))).float()  # Convertir [N,H,W,C] -> [N,C,H,W]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=settings.batch_size, seed=settings.SEED):
    """
    Create the dataloaders for train, validation, and test sets.
    """
    # Crear datasets
    train_dataset = NPYDataset(X_train, y_train)
    val_dataset   = NPYDataset(X_val, y_val)
    test_dataset  = NPYDataset(X_test, y_test)

    # Generador con semilla para reproducibilidad
    g = torch.Generator()
    g.manual_seed(seed)

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
