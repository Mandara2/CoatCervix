# data/dataloaders.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import settings

class NPYDataset(Dataset):
    """
    Dataset simple que recibe X e y en formato numpy arrays y devuelve tensores [C,H,W] y etiquetas.
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
    Crea los DataLoaders para train, val y test a partir de numpy arrays.
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
