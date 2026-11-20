import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Mortar Strength Predictor", layout="centered")

ARTIFACT_FILE = os.path.join(os.path.dirname(__file__), "best_mortar_model.pkl")

st.markdown(
    """
    <style>
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
    </style>
    """,
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
    d["SP/WBR"] = np.where(d["WBR"] > 0, d["SP"] / d["WBR"], np.nan)

    d["GGBS_CFA_Synergy"] = d["GGBS"] * d["CFA"]
    d["Slag_Hydration_Index"] = (d["GGBS"] * d["SiO2"]) / (d["WBR"] + eps)
    d["GGBS_CFA_Balance"] = d["GGBS"] / (d["CFA"] + 0.1)

    d = d.drop(columns=["Water", "FM"], errors="ignore")
    return d


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ARTIFACT_FILE)
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
        f"Sum of Local SHAP       : {phi.sum():+.2f} MPa",
        f"Final Prediction        : {pred_value:.2f} MPa",
        "-" * 68,
        "Top 10 Objective Local SHAP Feature Contributions:",
    ]
    for rank_i, (fname, fval, sval) in enumerate(ranked[:10], 1):
        direction = "[+]" if sval >= 0 else "[-]"
        lines.append(
            f"{rank_i:2d}. {fname:<22} ({fval:>7.3f}) -> {direction} {sval:+6.2f} MPa"
        )

    lines.append("-" * 68)
    if ood_warnings:
        lines.append("EXTRAPOLATION WARNING (Outside Training Domain - Interpret Carefully):")
        lines.extend(ood_warnings[:6])
    else:
        lines.append("[OK] All input & engineered features lie within the training domain.")

    return "\n".join(lines), bool(ood_warnings), ranked[:10]


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
    st.error(f"Artifact Not Found:\n{ARTIFACT_FILE}\n\nDetails:\n{e}")
    st.stop()

st.title("Cementless Mortar Strength Predictor")

input_fields = [
    ("WBR (Water/Binder):", "WBR", float(round(baseline["WBR"], 3)), "%.3f", 0.01),
    ("GGBS (fraction):", "GGBS", float(round(baseline["GGBS"], 3)), "%.3f", 0.01),
    ("CFA (fraction):", "CFA", float(round(baseline["CFA"], 3)), "%.3f", 0.01),
    ("FA (fraction):", "FA", float(round(baseline["FA"], 3)), "%.3f", 0.01),
    ("RUFA (fraction):", "RUFA", float(round(baseline["RUFA"], 3)), "%.3f", 0.01),
    ("SF (fraction):", "SF", float(round(baseline["SF"], 3)), "%.3f", 0.01),
    ("Aggregate ratio:", "Aggregate", float(round(baseline["Aggregate"], 3)), "%.3f", 0.05),
    ("Age (days):", "Age", float(int(round(baseline["Age"]))), "%.1f", 1.0),
    ("Al2O3 (%):", "Al2O3", float(round(baseline["Al2O3"], 2)), "%.2f", 0.1),
    ("SiO2 (%):", "SiO2", float(round(baseline["SiO2"], 2)), "%.2f", 0.1),
    ("SO3 (%):", "SO3", float(round(baseline["SO3"], 2)), "%.2f", 0.1),
    ("CaO (%):", "CaO", float(round(baseline["CaO"], 2)), "%.2f", 0.1),
    ("Fiber:", "Fiber", float(round(baseline["Fiber"], 4)), "%.4f", 0.001),
    ("Superplasticizer (SP):", "SP", float(round(baseline["SP"], 4)), "%.4f", 0.001),
]

with st.form("predictor_form"):
    raw_vals = {}
    col1, col2 = st.columns(2, gap="large")
    for i, (lbl_text, key, default_val, fmt, step_val) in enumerate(input_fields):
        target_col = col1 if i < 7 else col2
        raw_vals[key] = target_col.number_input(
            lbl_text, value=default_val, step=step_val, format=fmt
        )

    submitted = st.form_submit_button(
        "Predict Strength & Compute Local SHAP", type="primary"
    )

if submitted:
    try:
        # Convert oxide fractions (0-1) to wt% (0-100) only if all primary oxides are < 1
        if raw_vals["Al2O3"] < 1.0 and raw_vals["SiO2"] < 1.0 and raw_vals["CaO"] < 1.0:
            for ox in ("Al2O3", "SiO2", "SO3", "CaO"):
                raw_vals[ox] *= 100.0

        # Physical input validation
        if raw_vals["WBR"] <= 0 or raw_vals["WBR"] >= 2.0:
            raise ValueError("WBR must be strictly between 0 and 2.0.")
        if raw_vals["Age"] <= 0:
            raise ValueError("Age must be greater than 0 days.")
        if any(v < 0 for v in raw_vals.values()):
            raise ValueError("All mix proportions and oxide percentages must be non-negative.")

        nbinder = (
            raw_vals["GGBS"]
            + raw_vals["CFA"]
            + raw_vals["RUFA"]
            + raw_vals["SF"]
            + raw_vals["FA"]
        )
        if nbinder <= 0:
            raise ValueError("Sum of binder components (GGBS+CFA+RUFA+SF+FA) must be > 0.")

        df_input = apply_domain_features(pd.DataFrame([raw_vals]))[feature_names]

        pred_mpa = float(model_pipeline.predict(df_input)[0])
        report_text, is_ood, ranked_top10 = generate_objective_shap_report(
            explainer,
            model_name,
            feature_names,
            training_bounds,
            df_input,
            pred_mpa,
        )

        card_cls = "ood" if is_ood else "ok"
        status_text = (
            "[Warning: Input Outside Training Range]"
            if is_ood
            else "[OK: In-Distribution Prediction]"
        )

        st.markdown(
            f"""
            <div class="result-card {card_cls}">
                <div class="result-label">Predicted Strength ({model_name})</div>
                <div class="result-value">{pred_mpa:.2f} MPa</div>
                <span class="result-status {card_cls}">{status_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Real-Time Objective SHAP Breakdown:")

        max_abs = max((abs(s) for _, _, s in ranked_top10), default=1.0) or 1.0
        bars = []
        for fname, fval, sval in ranked_top10:
            pct = min(100.0, (abs(sval) / max_abs) * 100.0)
            bar_cls = "shap-pos" if sval >= 0 else "shap-neg"
            val_color = "#10b981" if sval >= 0 else "#ef4444"
            bars.append(
            )
        st.markdown(f'<div class="shap-card">{"".join(bars)}</div>', unsafe_allow_html=True)

        st.code(report_text, language="text")

    except ValueError as err:
        st.error(f"Invalid input:\n{err}")
    except KeyError as err:
        st.error(
            f"Missing feature {err}. Re-run Cell 8 to regenerate best_mortar_model.pkl."
        )
