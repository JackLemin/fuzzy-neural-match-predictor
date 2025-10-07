# app.py (Streamlit)
import streamlit as st
import numpy as np
from fuzzy import fuzzy_preprocess, explanation_from_fuzzy
from model import load_model, prediction_to_label
from utils import compute_confidence
import torch

st.set_page_config(page_title="Fuzzy Neural Match Predictor", layout="centered")

st.title("Fuzzy Neural Match Predictor (FNMP)")
st.markdown("Hybrid neuro-fuzzy model for match outcome prediction. Enter match features below:")

with st.form("input_form"):
    team_form = st.slider("Team Form (0 to 1)", 0.0, 1.0, 0.8, 0.01)
    player_form = st.slider("Player Form (0 to 1)", 0.0, 1.0, 0.9, 0.01)
    weather = st.slider("Weather favorability (0 to 1)", 0.0, 1.0, 0.7, 0.01)
    injuries = st.slider("Injuries (0 = few injuries .. 1 = many injuries)", 0.0, 1.0, 0.2, 0.01)
    xg = st.slider("xG (0 to 1)", 0.0, 1.0, 0.85, 0.01)
    importance = st.slider("Match Importance (0 to 1)", 0.0, 1.0, 0.9, 0.01)

    submitted = st.form_submit_button("Predict")

if submitted:
    sample = [team_form, player_form, weather, injuries, xg, importance]
    pre = fuzzy_preprocess(sample)
    explanation = explanation_from_fuzzy(sample, pre)

    # Load model
    try:
        model, md = load_model("models/fnmp_model.pt")
    except Exception as e:
        st.error("Model not found. Please run `train.py` first to create the model.")
        st.stop()

    model.eval()
    with torch.no_grad():
        x = torch.tensor(pre[np.newaxis, :], dtype=torch.float32)
        out = model(x).cpu().numpy().squeeze()
    label = prediction_to_label(out)
    conf = compute_confidence(out)

    label_map = {-1: "Loss", 0: "Draw", 1: "Win"}
    st.subheader("Result")
    st.metric("Predicted Outcome", f"{label_map[int(label)]} (raw {out:.3f})", delta=f"Confidence {conf*100:.1f}%")
    st.write("### Fuzzy Explanation")
    st.write(explanation)
    st.write("### Notes")
    st.write("- Output is in [-1,1]; thresholds map to Win/Draw/Loss.")
    st.write("- Fuzzy rules are used to convert noisy inputs to linguistic-like features.")
