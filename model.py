# model.py
import torch
import torch.nn as nn
import numpy as np
import os

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=16, output_dim=1, activation='tanh'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # output in [-1,1] to match labels { -1, 0, 1 }
        )

    def forward(self, x):
        return self.net(x)

def save_model(model, path="models/fnmp_model.pt", metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"model_state": model.state_dict(), "metadata": metadata}
    torch.save(state, path)

def load_model(path="models/fnmp_model.pt", device=None, input_dim=6, hidden_dim=16):
    if device is None:
        device = torch.device("cpu")
    data = torch.load(path, map_location=device)
    md = data.get("metadata", {})
    model = FeedForwardNN(input_dim=md.get("input_dim", input_dim),
                          hidden_dim=md.get("hidden_dim", hidden_dim),
                          activation=md.get("activation", "tanh"))
    model.load_state_dict(data["model_state"])
    model.to(device)
    model.eval()
    return model, md

def prediction_to_label(pred):
    """
    pred: float or numpy array in [-1,1] -> convert to categorical {-1,0,1}
    thresholding: [-0.25, 0.25] -> draw (0) (tunable)
    """
    p = np.asarray(pred)
    labels = np.ones_like(p, dtype=int) * -1
    labels[p > 0.25] = 1
    labels[np.abs(p) <= 0.25] = 0
    return labels
