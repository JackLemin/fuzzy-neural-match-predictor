# fuzzy.py
import numpy as np

def fuzzy_preprocess(sample):
    """
    sample: 1D array-like of length 6:
      [team_form, player_form, weather, injuries, xG, match_importance]
    returns: numpy array of preprocessed fuzzy features (floats 0-1)
    """
    s = np.asarray(sample, dtype=np.float32)
    p = np.zeros_like(s)

    # Team form
    if s[0] > 0.8:
        p[0] = 1.0
    elif s[0] > 0.6:
        p[0] = 0.75
    elif s[0] > 0.4:
        p[0] = 0.5
    else:
        p[0] = 0.25

    # Player form
    if s[1] > 0.8:
        p[1] = 1.0
    elif s[1] > 0.6:
        p[1] = 0.75
    elif s[1] > 0.4:
        p[1] = 0.5
    else:
        p[1] = 0.25

    # Weather
    if s[2] > 0.7:
        p[2] = 1.0
    elif s[2] > 0.4:
        p[2] = 0.5
    else:
        p[2] = 0.0

    # Injuries (lower = good)
    if s[3] < 0.2:
        p[3] = 1.0
    elif s[3] < 0.5:
        p[3] = 0.5
    else:
        p[3] = 0.0

    # xG: used as-is (assumed normalized)
    p[4] = np.clip(s[4], 0.0, 1.0)

    # Match importance
    if s[5] > 0.8:
        p[5] = 1.0
    elif s[5] > 0.6:
        p[5] = 0.75
    elif s[5] > 0.4:
        p[5] = 0.5
    else:
        p[5] = 0.25

    return p

def explanation_from_fuzzy(sample, preprocessed):
    """
    Create a human-readable explanation that uses fuzzy rules and feature contributions.
    """
    labels = ["Team Form", "Player Form", "Weather", "Injuries", "xG", "Match Importance"]
    descs = []
    s = np.asarray(sample)
    p = np.asarray(preprocessed)

    # Short linguistic mapping
    mapping = {
        1.0: "Excellent/Very favorable",
        0.75: "High",
        0.5: "Medium/Neutral",
        0.25: "Low",
        0.0: "Adverse/Very unfavorable"
    }

    for i, lab in enumerate(labels):
        v = p[i]
        descs.append(f"{lab}: {mapping.get(float(v), f'{v:.2f}')}")
    expl = "; ".join(descs)
    return expl
