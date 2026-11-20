import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Mortar Strength Predictor")

ARTIFACT_FILE = os.path.join(os.path.dirname(__file__), "best_mortar_model.pkl")


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
    st.error(f"Artifact Not Found:\n{ARTIFACT_FILE}\n\nDetails:\n{e}")
    st.stop()

st.title(f"Cementless Mortar Strength Predictor")

input_fields = [
    ("WBR (Water/Binder):", "WBR", float(round(baseline["WBR"], 3)), "%.3f"),
    ("GGBS (fraction):", "GGBS", float(round(baseline["GGBS"], 3)), "%.3f"),
    ("CFA (fraction):", "CFA", float(round(baseline["CFA"], 3)), "%.3f"),
    ("FA (fraction):", "FA", float(round(baseline["FA"], 3)), "%.3f"),
    ("RUFA (fraction):", "RUFA", float(round(baseline["RUFA"], 3)), "%.3f"),
    ("SF (fraction):", "SF", float(round(baseline["SF"], 3)), "%.3f"),
    ("Aggregate ratio:", "Aggregate", float(round(baseline["Aggregate"], 3)), "%.3f"),
    ("Age (days):", "Age", float(int(round(baseline["Age"]))), "%.1f"),
    ("Al2O3 (%):", "Al2O3", float(round(baseline["Al2O3"], 2)), "%.2f"),
    ("SiO2 (%):", "SiO2", float(round(baseline["SiO2"], 2)), "%.2f"),
    ("SO3 (%):", "SO3", float(round(baseline["SO3"], 2)), "%.2f"),
    ("CaO (%):", "CaO", float(round(baseline["CaO"], 2)), "%.2f"),
    ("Fiber:", "Fiber", float(round(baseline["Fiber"], 4)), "%.4f"),
    ("Superplasticizer (SP):", "SP", float(round(baseline["SP"], 4)), "%.4f"),
]

with st.form("predictor_form"):
    raw_vals = {}
    col1, col2 = st.columns(2)
    for i, (lbl_text, key, default_val, fmt) in enumerate(input_fields):
        target_col = col1 if i < 7 else col2
        raw_vals[key] = target_col.number_input(lbl_text, value=default_val, format=fmt)

    submitted = st.form_submit_button("Predict Strength & Compute Local SHAP")

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
        report_text, is_ood = generate_objective_shap_report(
            explainer,
            model_name,
            feature_names,
            training_bounds,
            df_input,
            pred_mpa,
        )

        if is_ood:
            st.error(
                f"Predicted Strength ({model_name}): {pred_mpa:.2f} MPa\n\n"
                "[Warning: Input Outside Training Range]"
            )
        else:
            st.success(
                f"Predicted Strength ({model_name}): {pred_mpa:.2f} MPa\n\n"
                "[OK: In-Distribution Prediction]"
            )

        st.subheader("Real-Time Objective SHAP Breakdown:")
        st.code(report_text, language="text")

    except ValueError as err:
        st.error(f"Invalid input:\n{err}")
    except KeyError as err:
        st.error(
            f"Missing feature {err}. Re-run Cell 8 to regenerate best_mortar_model.pkl."
        )
