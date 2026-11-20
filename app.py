import os
import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
import pandas as pd
import shap

# Update to your artifact folder (defaults to current working directory)
FOLDER_PATH = r"C:\Users\a0338\Desktop\ML-Mortar"
ARTIFACT_FILE = os.path.join(FOLDER_PATH, "best_mortar_model.pkl")


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


try:
    bundle = joblib.load(ARTIFACT_FILE)
    model_pipeline = bundle["model"]
    model_name = bundle["model_name"]
    feature_names = bundle["feature_columns"]
    baseline = bundle["baseline_medians"]
    training_bounds = bundle["training_bounds"]
    shap_background = bundle["shap_background"]
except FileNotFoundError as e:
    root_err = tk.Tk()
    root_err.withdraw()
    messagebox.showerror(
        "Artifact Not Found",
        f"Run Cell 8 first to generate:\n{ARTIFACT_FILE}\n\nDetails:\n{e}",
    )
    raise SystemExit()

# Build objective SHAP explainer once at startup
if model_name in ("Random Forest", "XGBoost"):
    explainer = shap.TreeExplainer(model_pipeline.named_steps["model"])
else:
    explainer = shap.Explainer(model_pipeline.predict, shap_background)


def generate_objective_shap_report(df_input: pd.DataFrame, pred_value: float):
    """Computes real-time local SHAP values and checks training-range extrapolation."""
    sv = explainer(df_input)
    base_val = float(np.asarray(sv.base_values).ravel()[0])
    phi = np.asarray(sv.values).ravel()

    # Check out-of-distribution inputs (purely informational; never modifies prediction)
    ood_warnings = []
    for col in feature_names:
        val = float(df_input[col].iloc[0])
        lo, hi = training_bounds[col]["min"], training_bounds[col]["max"]
        if val < lo or val > hi:
            ood_warnings.append(f"  ! {col} = {val:.4g} (training range: [{lo:.4g}, {hi:.4g}])")

    # Sort features by absolute local SHAP contribution
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
        lines.append(f"{rank_i:2d}. {fname:<22} ({fval:>7.3f}) -> {direction} {sval:+6.2f} MPa")

    if ood_warnings:
        lines.append("-" * 68)
        lines.append("EXTRAPOLATION WARNING (Outside Training Domain — Interpret Carefully):")
        lines.extend(ood_warnings[:6])
    else:
        lines.append("-" * 68)
        lines.append("[✓] All input & engineered features lie within the training domain.")

    return "\n".join(lines), bool(ood_warnings)


def calculate_strength():
    try:
        raw_vals = {
            "WBR": float(entries["WBR"].get()),
            "GGBS": float(entries["GGBS"].get()),
            "FA": float(entries["FA"].get()),
            "CFA": float(entries["CFA"].get()),
            "RUFA": float(entries["RUFA"].get()),
            "SF": float(entries["SF"].get()),
            "Aggregate": float(entries["Aggregate"].get()),
            "Age": float(entries["Age"].get()),
            "Al2O3": float(entries["Al2O3"].get()),
            "SiO2": float(entries["SiO2"].get()),
            "SO3": float(entries["SO3"].get()),
            "CaO": float(entries["CaO"].get()),
            "Fiber": float(entries["Fiber"].get()),
            "SP": float(entries["SP"].get()),
        }

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

        nbinder = raw_vals["GGBS"] + raw_vals["CFA"] + raw_vals["RUFA"] + raw_vals["SF"] + raw_vals["FA"]
        if nbinder <= 0:
            raise ValueError("Sum of binder components (GGBS+CFA+RUFA+SF+FA) must be > 0.")

        df_input = apply_domain_features(pd.DataFrame([raw_vals]))[feature_names]

        # Pure, unmodified model prediction
        pred_mpa = float(model_pipeline.predict(df_input)[0])
        report_text, is_ood = generate_objective_shap_report(df_input, pred_mpa)

        if is_ood:
            label_result.config(
                text=f"Predicted Strength ({model_name}): {pred_mpa:.2f} MPa\n[Warning: Input Outside Training Range]",
                fg="#d9534f",
            )
        else:
            label_result.config(
                text=f"Predicted Strength ({model_name}): {pred_mpa:.2f} MPa\n[✓ In-Distribution Prediction]",
                fg="#1b7837",
            )

        text_advisory.config(state="normal")
        text_advisory.delete("1.0", tk.END)
        text_advisory.insert(tk.END, report_text)
        text_advisory.config(state="disabled")

    except ValueError as err:
        messagebox.showerror("Input Error", f"Invalid input:\n{err}")
    except KeyError as err:
        messagebox.showerror(
            "Feature Mismatch",
            f"Missing feature {err}. Re-run Cell 8 to regenerate best_mortar_model.pkl.",
        )


if __name__ == "__main__":
    root = tk.Tk()
    root.title(f"Mortar Strength Predictor — Objective {model_name} + Real-Time SHAP")
    root.geometry("580x820")
    root.resizable(False, False)

    input_fields = [
        ("WBR (Water/Binder):", "WBR", round(baseline["WBR"], 3)),
        ("GGBS (fraction):", "GGBS", round(baseline["GGBS"], 3)),
        ("CFA (fraction):", "CFA", round(baseline["CFA"], 3)),
        ("FA (fraction):", "FA", round(baseline["FA"], 3)),
        ("RUFA (fraction):", "RUFA", round(baseline["RUFA"], 3)),
        ("SF (fraction):", "SF", round(baseline["SF"], 3)),
        ("Aggregate ratio:", "Aggregate", round(baseline["Aggregate"], 3)),
        ("Age (days):", "Age", int(round(baseline["Age"]))),
        ("Al2O3 (%):", "Al2O3", round(baseline["Al2O3"], 2)),
        ("SiO2 (%):", "SiO2", round(baseline["SiO2"], 2)),
        ("SO3 (%):", "SO3", round(baseline["SO3"], 2)),
        ("CaO (%):", "CaO", round(baseline["CaO"], 2)),
        ("Fiber:", "Fiber", round(baseline["Fiber"], 4)),
        ("Superplasticizer (SP):", "SP", round(baseline["SP"], 4)),
    ]

    entries = {}
    for i, (lbl_text, key, default_val) in enumerate(input_fields):
        tk.Label(root, text=lbl_text, anchor="w", font=("Arial", 9)).grid(
            row=i, column=0, padx=18, pady=3, sticky="w"
        )
        ent = tk.Entry(root, width=22, font=("Arial", 9))
        ent.insert(0, str(default_val))
        ent.grid(row=i, column=1, padx=18, pady=3, sticky="e")
        entries[key] = ent

    tk.Button(
        root,
        text="Predict Strength & Compute Local SHAP",
        command=calculate_strength,
        bg="#007acc",
        fg="white",
        font=("Arial", 10, "bold"),
        padx=12,
        pady=5,
    ).grid(row=len(input_fields), column=0, columnspan=2, pady=10)

    label_result = tk.Label(
        root,
        text="Predicted Strength: -- MPa",
        font=("Arial", 11, "bold"),
        justify="center",
    )
    label_result.grid(row=len(input_fields) + 1, column=0, columnspan=2, pady=4)

    tk.Label(
        root,
        text="Real-Time Objective SHAP Breakdown:",
        font=("Arial", 9, "bold"),
        anchor="w",
    ).grid(row=len(input_fields) + 2, column=0, columnspan=2, padx=18, pady=(6, 2), sticky="w")

    text_advisory = tk.Text(
        root,
        width=72,
        height=18,
        font=("Consolas", 8),
        bg="#f8f9fa",
        fg="#212529",
        relief="solid",
        bd=1,
        state="disabled",
    )
    text_advisory.grid(row=len(input_fields) + 3, column=0, columnspan=2, padx=18, pady=4)

    root.mainloop()
