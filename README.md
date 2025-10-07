# Fuzzy Neural Match Predictor (FNMP)

Hybrid neuro-fuzzy system for predicting match outcomes. Converts raw sports features into fuzzy linguistic values and feeds them to a small neural network. Includes hyperparameter autotuning (Optuna), model saving, explainable fuzzy explanations, and a Streamlit app for live predictions.

## Highlights
- Fuzzy preprocessing for human-like reasoning.
- PyTorch feed-forward model trained end-to-end.
- Optuna-based auto hyperparameter tuning.
- Simple rule-based explanations + confidence score.
- Streamlit app for interactive predictions.

## Setup
1. Create virtual env and install:
```bash
pip install -r requirements.txt
```

2. Train and auto-tune (creates `models/fnmp_model.pt`):
```bash
python train.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Quick CLI prediction:
```bash
python predict_cli.py
```

## Files
- `train.py` — training + optuna tuning
- `model.py` — PyTorch model + save/load
- `fuzzy.py` — fuzzy preprocessing & natural language explanation
- `app.py` — Streamlit UI
- `data.py` — example toy dataset (from original MATLAB)
- `utils.py` — helpers (confidence, permutation importance)
