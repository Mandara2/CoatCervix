# data/load_data.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import settings

def load_data():
    """
    upload the data from .npy files,
    and generate a validation split from the training data.
    """
    X_train_full = np.load(os.path.join(settings.DATA_PATH, "X_train.npy"))
    y_train_full = np.load(os.path.join(settings.DATA_PATH, "y_train.npy"))
    X_test  = np.load(os.path.join(settings.DATA_PATH, "X_test.npy"))
    y_test  = np.load(os.path.join(settings.DATA_PATH, "y_test.npy"))

    # Create
    X_train, X_val, y_train, y_val = split_train_val(X_train_full, y_train_full)

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_train_val(X, y, test_size=0.2, random_state=42):
    """
    generate a validation split from the training data, using stratified sampling to maintain class distribution.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train_split, X_val_split, y_train_split, y_val_split
