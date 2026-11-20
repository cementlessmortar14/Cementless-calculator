import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Mortar Strength Predictor", layout="centered")

ARTIFACT_FILE = os.path.join(os.path.dirname(__file__), "best_mortar_model.pkl")

st.markdown(
    """<style>
.block-container {
    max-width: 860px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
h1 {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 1.25rem !important;
}
h3 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
}
div[data-testid="stForm"] {
    border-radius: 14px;
    padding: 1.5rem 1.75rem 1.75rem 1.75rem;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background-color: var(--secondary-background-color);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
}
div[data-testid="stNumberInput"] label {
    font-size: 0.88rem;
    font-weight: 500;
    opacity: 0.9;
}
div[data-testid="stNumberInput"] input {
    border-radius: 8px;
}
div[data-testid="stFormSubmitButton"] {
    margin-top: 1rem;
}
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    border-radius: 10px;
    padding: 0.65rem 1.25rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: transform 0.12s ease, opacity 0.12s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
}
.result-card {
    margin-top: 1.5rem;
    padding: 1.5rem 1.75rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.22);
}
.result-card.ok {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.10), rgba(15, 118, 110, 0.04));
    border-color: rgba(16, 185, 129, 0.38);
}
.result-card.ood {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.10), rgba(245, 158, 11, 0.04));
    border-color: rgba(239, 68, 68, 0.42);
}
.result-label {
    font-size: 0.92rem;
    font-weight: 600;
    opacity: 0.8;
}
.result-value {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0.25rem 0 0.5rem 0;
}
.result-status {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}
.result-status.ok {
    background: rgba(16, 185, 129, 0.16);
    color: #10b981;
}
.result-status.ood {
    background: rgba(239, 68, 68, 0.16);
    color: #ef4444;
}
.shap-card {
    padding: 1.25rem 1.5rem;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    background-color: var(--secondary-background-color);
    margin-bottom: 1rem;
}
.shap-row {
    margin-bottom: 0.65rem;
}
.shap-row:last-child {
    margin-bottom: 0;
}
.shap-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    margin-bottom: 0.22rem;
}
.shap-name {
    font-weight: 600;
}
.shap-fval {
    opacity: 0.65;
    font-size: 0.82rem;
    margin-left: 0.35rem;
}
.shap-track {
    width: 100%;
    height: 7px;
    border-radius: 999px;
    background: rgba(128, 128, 128, 0.16);
    overflow: hidden;
}
.shap-pos {
    height: 100%;
    border-radius: 999px;
    background: #10b981;
}
.shap-neg {
    height: 100%;
    border-radius: 999px;
    background: #ef4444;
}
</style>""",
    unsafe_allow_html=True,
)


def apply_domain_features(data: pd.DataFrame) -> pd.DataFrame:
    """Identical deterministic feature engineering used during training."""
    d = data.copy()
    d["nBinder"] = d["GGBS"] + d["CFA"] + d["RUFA"] + d["SF"] + d["FA"]
    d["nBinder"] = d["nBinder"].replace(0, np.nan)

    if "Water" in d.columns:
        d["WBR"] = d["Water"] / d["nBinder"]
    if "Aggregate" in d.columns:
        d["ABR"] = d["Aggregate"] / d["nBinder"]

    eps = 1e-5
    d["SiO2/Al2O3"] = d["SiO2"] / (d["Al2O3"] + eps)
    d["CaO/Al2O3"] = d["CaO"] / (d["Al2O3"] + eps)
    d["CaO/SiO2"] = d["CaO"] / (d["SiO2"] + eps)
    d["SO3/CaO"] = d["SO3"] / (d["CaO"] + eps)
    d["SP/WBR"] = np.where(d["WBR"] > 0, d["SP"] / d["W
