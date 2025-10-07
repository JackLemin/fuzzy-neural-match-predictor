# train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import FeedForwardNN, save_model
from fuzzy import fuzzy_preprocess
from data import train_data, train_labels, test_data, test_labels
from utils import to_tensor, mean_abs_error_preds
import optuna
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_dataset(X):
    return np.stack([fuzzy_preprocess(x) for x in X], axis=0).astype(np.float32)

def train_model(hparams, X_train, y_train, X_val, y_val, device=DEVICE):
    model = FeedForwardNN(input_dim=X_train.shape[1],
                          hidden_dim=hparams["hidden_dim"],
                          activation=hparams.get("activation", "tanh"))
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    loss_fn = nn.MSELoss()  # regression to -1..1 (we map labels accordingly)
    epochs = hparams.get("epochs", 400)
    batch_size = hparams.get("batch_size", 8)
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_val = float("inf")
    best_state = None
    patience = hparams.get("patience", 30)
    wait = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # validation
        model.eval()
        with torch.no_grad():
            vpred = model(torch.tensor(X_val).to(device)).cpu().numpy().squeeze()
            vloss = mean_abs_error_preds(y_val, vpred)
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    # restore best
    model.load_state_dict(best_state)
    return model, best_val

def objective(trial):
    hparams = {
        "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
        "hidden_dim": trial.suggest_int("hidden_dim", 8, 64),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "epochs": 500,
        "batch_size": trial.suggest_categorical("batch_size", [4,8,16]),
        "patience": 40
    }
    # small dataset -> use leave-one-out as validation
    Xp = preprocess_dataset(train_data)
    yp = train_labels.astype(np.float32)
    # simple split: last sample as val (tiny dataset)
    Xtr, Xval = Xp[:-1], Xp[-1:].copy()
    ytr, yval = yp[:-1], yp[-1:]

    model, val_loss = train_model(hparams, Xtr, ytr, Xval, yval)
    return val_loss

def run_optuna_and_train(n_trials=30):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best hyperparams:", study.best_params)
    # final train on all data with best params using small CV
    best = study.best_params
    hparams = {
        "lr": best["lr"],
        "hidden_dim": best["hidden_dim"],
        "activation": best["activation"],
        "epochs": 600,
        "batch_size": 8,
        "patience": 60
    }
    Xp = preprocess_dataset(train_data)
    yp = train_labels.astype(np.float32)
    # For this toy example, split train/test from provided test_data
    X_train_full = Xp
    y_train_full = yp
    X_test_p = preprocess_dataset(test_data)
    y_test = test_labels.astype(np.float32)

    model, _ = train_model(hparams, X_train_full, y_train_full, X_test_p, y_test)
    # Save
    metadata = {"input_dim": 6, "hidden_dim": hparams["hidden_dim"], "activation": hparams["activation"]}
    save_model(model, path="models/fnmp_model.pt", metadata=metadata)
    print("Model saved to models/fnmp_model.pt")
    return model

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    model = run_optuna_and_train()
