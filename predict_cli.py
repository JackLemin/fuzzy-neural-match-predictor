# predict_cli.py
import numpy as np
import torch
from fuzzy import fuzzy_preprocess, explanation_from_fuzzy
from model import load_model, prediction_to_label
from utils import compute_confidence

def predict(sample):
    p = fuzzy_preprocess(sample)
    model, md = load_model("models/fnmp_model.pt")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(p[np.newaxis, :], dtype=torch.float32)
        out = model(x).cpu().numpy().squeeze()
    label = prediction_to_label(out)
    conf = compute_confidence(out)
    explanation = explanation_from_fuzzy(sample, p)
    return {"raw": float(out), "label": int(label), "confidence": conf, "explanation": explanation}

if __name__ == "__main__":
    # example input (team_form, player_form, weather, injuries, xG, importance)
    sample = [0.8, 0.9, 0.7, 0.2, 0.85, 0.9]
    result = predict(sample)
    print("Prediction:", result)
