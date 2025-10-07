# utils.py
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
import torch

def mean_abs_error_preds(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def compute_confidence(score):
    """
    Convert continuous output in [-1,1] to a confidence [0,1]
    Higher absolute value -> higher confidence
    """
    return float(min(1.0, abs(score)))

def to_tensor(x, device=None):
    t = torch.tensor(x, dtype=torch.float32)
    if device:
        t = t.to(device)
    return t

def permutation_importance_predict_fn(model, X_pre, baseline_metric_fn, n_repeats=30, random_state=0):
    """
    Very small permutation importance: returns importance for each feature by
    measuring metric degradation when column values are shuffled.
    model: a PyTorch model that returns a scalar prediction when fed a 2D torch tensor
    X_pre: numpy array (n_samples, n_features)
    baseline_metric_fn: function(X -> baseline metric, e.g., accuracy over some ground truth)
    """
    rng = np.random.RandomState(random_state)
    n_features = X_pre.shape[1]
    importances = np.zeros(n_features)
    baseline_score = baseline_metric_fn(X_pre)
    for f in range(n_features):
        scores = []
        for _ in range(n_repeats):
            Xs = X_pre.copy()
            rng.shuffle(Xs[:, f])
            scores.append(baseline_metric_fn(Xs))
        # importance = baseline - mean(shuffled)
        importances[f] = baseline_score - np.mean(scores)
    return importances, baseline_score
