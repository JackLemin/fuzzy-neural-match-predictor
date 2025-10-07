# data.py
import numpy as np

# Training data (5 samples, 6 features)
train_data = np.array([
    [0.7, 0.8, 0.6, 0.3, 0.75, 0.8],
    [0.6, 0.5, 0.4, 0.4, 0.5, 0.6],
    [0.9, 0.7, 0.7, 0.2, 0.9, 1.0],
    [0.8, 0.8, 0.7, 0.1, 0.85, 0.9],
    [0.5, 0.4, 0.5, 0.5, 0.45, 0.5],
], dtype=np.float32)

# Labels: 1 (win), -1 (loss)
train_labels = np.array([1, -1, 1, 1, -1], dtype=np.float32)

# Testing data (5 samples)
test_data = np.array([
    [0.8, 0.9, 0.7, 0.2, 0.85, 0.9],
    [0.5, 0.6, 0.4, 0.5, 0.5, 0.7],
    [0.3, 0.5, 0.6, 0.4, 0.6, 0.6],
    [0.9, 0.8, 0.8, 0.1, 0.9, 1.0],
    [0.4, 0.3, 0.5, 0.7, 0.3, 0.5],
], dtype=np.float32)

test_labels = np.array([1, -1, 0, 1, -1], dtype=np.float32)  # keep 0 for draw where known
