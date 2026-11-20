import io
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

GITHUB_RAW_URLS = [
    "https://raw.githubusercontent.com/cementlessmortar14/Cementless-calculator/main/best_mortar_model.pkl",
    "https://raw.githubusercontent.com/cementlessmortar14/Cementless-calculator/master/best_mortar_model.pkl",
]
ARTIFACT_FILE = Path(__file__).resolve().parent / "best_mortar_model.pkl"


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
    d["SP/WBR"] = np.where(d["WBR"] > 0, d["SP"] / d["WBR"], np.nan)

    d["GGBS_CFA_Synergy"] = d["GGBS"] * d["CFA"]
    d["Slag_Hydration_Index"] = (d["GGBS"] * d["SiO2"]) / (d["WBR"] + eps)
    d["GGBS_CFA_Balance"] = d["GGBS"] / (d["CFA"] + 0.1)

    d = d.drop(columns=["Water", "FM"], errors="ignore")
    return d


@st.cache_resource
def load_artifacts():
    bundle = None
    if ARTIFACT_FILE.exists():
        bundle = joblib.load(ARTIFACT_FILE)
    else:
        last_err = None
        for url in GITHUB_RAW_URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    bundle = joblib.load(io.BytesIO(resp.read()))
                break
            except Exception as err:
                last_err = err
        if bundle is None:
            raise FileNotFoundError(
                f"Could not find {ARTIFACT_FILE} locally or on GitHub. Details: {last_err}"
            )

    model_pipeline = bundle["model"]
    model_name = bundle["model_name"]
    feature_names = bundle["feature_columns"]
    baseline = bundle["baseline_medians"]
    training_bounds = bundle["training_bounds"]
    shap_background = bundle["shap_background"]

    if model_name in ("Random Forest", "XGBoost"):
        explainer = shap.TreeExplainer(model_pipeline.named_steps["model"])
    else:
        explainer = shap.Explainer(model_pipeline.predict, shap_background)

    return (
        model_pipeline,
        model_name,
        feature_names,
        baseline,
        training_bounds,
        explainer,
    )


def generate_objective_shap_report(
    explainer,
    model_name: str,
    feature_names: list,
    training_bounds: dict,
    df_input: pd.DataFrame,
    pred_value: float,
):
    """Computes real-time local SHAP values and checks training-range extrapolation."""
    sv = explainer(df_input)
    base_val = float(np.asarray(sv.base_values).ravel()[0])
    phi = np.asarray(sv.values).ravel()

    ood_warnings = []
    for col in feature_names:
        val = float(df_input[col].iloc[0])
        lo, hi = training_bounds[col]["min"], training_bounds[col]["max"]
        if val < lo or val > hi:
            ood_warnings.append(
                f"  ! {col} = {val:.4g} (training range: [{lo:.4g}, {hi:.4g}])"
            )

    ranked = sorted(
        zip(feature_names, df_input.iloc[0].values, phi),
        key=lambda x: abs(x[2]),
        reverse=True,
    )

    lines = [
        f"Model: {model_name} (Unmodified ML Prediction)",
        f"SHAP Base Value E[f(x)] : {base_val:.2f} MPa",
        f"Sum of Local SHAP (Σφ)  : {phi.sum():+.2f} MPa",
        f"Final Prediction        : {pred_value:.2f} MPa",
        "-" * 68,
        "Top 10 Objective Local SHAP Feature Contributions:",
    ]
    for rank_i, (fname, fval, sval) in enumerate(ranked[:10], 1):
        direction = "[+]" if sval >= 0 else "[-]"
        lines.append(
            f"{rank_i:2d}. {fname:<22} ({fval:>7.3f}) -> {direction} {sval:+6.2f} MPa"
        )

    if ood_warnings:
        lines.append("-" * 68)
        lines.append("EXTRAPOLATION WARNING (Outside Training Domain — Interpret Carefully):")
        lines.extend(ood_warnings[:6])
    else:
        lines.append("-" * 68)
        lines.append("[✓] All input & engineered features lie within the training domain.")

    return "\n".join(lines), bool(ood_warnings)


try:
    (
        model_pipeline,
        model_name,
        feature_names,
        baseline,
        training_bounds,
        explainer,
    ) = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Artifact Not Found:\n\n{e}")
    st.stop()

st.set_page_config(
    page_title=f"Mortar Strength Predictor — Objective {model_name} + Real-Time SHAP"
)
st.title(f"Mort
