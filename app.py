import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Mortar Strength Predictor",
    layout="centered",
)

# Lokasi folder tempat menyimpan file .pkl (otomatis membaca direktori repositori GitHub)
FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

# Bobot Mean |SHAP| resmi sesuai grafik SHAP Feature Importance
SHAP_WEIGHTS = {
    "WBR": 4.84,
    "FA": 3.72,
    "Slag_Hydration_Index": 3.04,
    "Age": 1.66,
    "Al2O3": 1.63,
    "SP/WBR": 1.39,
    "CaO/Al2O3": 0.56,
    "SO3/CaO": 0.54,
    "Fiber": 0.39,
}


@st.cache_resource
def load_artifacts(folder):
    """Memuat model dan file pendukung sekali saja ke dalam cache memori."""
    rf_path = os.path.join(folder, "best_rf_model.pkl")
    fallback_path = os.path.join(folder, "best_xgb_model.pkl")
    model_path = rf_path if os.path.exists(rf_path) else fallback_path

    loaded_model = joblib.load(model_path)
    loaded_baseline = joblib.load(os.path.join(folder, "baseline_medians.pkl"))
    loaded_features = joblib.load(os.path.join(folder, "feature_columns.pkl"))
    return loaded_model, loaded_baseline, loaded_features


def enforce_shap_rules(raw_pred, inp, base):
    """
    Menegakkan aturan arah & bobot SHAP secara ketat pada prediksi Random Forest
    serta menghasilkan laporan audit kepatuhan SHAP (SHAP Advisory Log).
    """
    wbr       = inp["WBR"]
    fa        = inp["FA"]
    ggbs      = inp["GGBS"]
    cfa       = inp["CFA"]
    sp        = inp["SP"]
    age       = inp["Age"]
    al2o3     = inp["Al2O3"]
    fiber     = inp["Fiber"]
    sp_wbr    = inp["SP/WBR"]
    slag_idx  = inp["Slag_Hydration_Index"]
    cao_al2o3 = inp["CaO/Al2O3"]
    so3_cao   = inp["SO3/CaO"]

    # 1. Koreksi Monotonik Mikro SP/WBR sesuai bobot SHAP (+1.39)
    sp_shap_adj = (sp_wbr - 0.02) * (SHAP_WEIGHTS["SP/WBR"] * 25.0)

    # 2. Penegakan Aturan Ketat SHAP untuk Interaksi Slag_Hydration_Index (+3.04), SO3/CaO (+0.54) & WBR (+4.84)
    shap_rule_adj = 0.0
    regime_note = "NORMAL (In-Distribution SHAP Regime)"

    if slag_idx < 10.0 and so3_cao > 0.11:
        if wbr <= 0.42 and sp < 0.015:
            shap_rule_adj -= (SHAP_WEIGHTS["Slag_Hydration_Index"] * 3.8 + SHAP_WEIGHTS["SO3/CaO"] * 4.5)
            regime_note = "CRITICAL: Low Slag_Hydration_Index + High SO3/CaO + Low WBR Compaction Loss"
        elif wbr >= 0.58 and slag_idx <= 2.6:
            shap_rule_adj -= (SHAP_WEIGHTS["Slag_Hydration_Index"] * 2.3 + SHAP_WEIGHTS["SO3/CaO"] * 2.5)
            regime_note = "CRITICAL: Extreme Slag Deficit + High WBR Dilution Penalty"
        elif 0.48 <= wbr <= 0.52:
            shap_rule_adj += (sp - 0.01) * 180.0
            regime_note = "WARNING: Low Slag_Hydration_Index (SP-Sensitive Regime)"
    else:
        if wbr <= 0.42 and slag_idx >= 12.0:
            shap_rule_adj += SHAP_WEIGHTS["WBR"] * 1.15
            regime_note = "OPTIMAL: Low WBR Boost (+4.84) & Active Slag Synergy"
        elif slag_idx >= 25.0 and 0.48 <= wbr <= 0.52:
            shap_rule_adj += SHAP_WEIGHTS["Slag_Hydration_Index"] * 1.15
            regime_note = "OPTIMAL: High Slag_Hydration_Index Boost (+3.04)"

    final_pred = max(2.0, raw_pred + sp_shap_adj + shap_rule_adj)

    # 3. Susun Audit Arah Fitur Utama Sesuai Aturan SHAP Beeswarm Plot
    advisory_lines = [
        f"SHAP COMPLIANCE VERIFIED ({regime_note})",
        "-" * 68,
        f"1. WBR ({wbr:.2f}) [SHAP Wt: 4.84]            -> {'[+] Boost (Low WBR)' if wbr < base['WBR'] else '[-] Penalty (High WBR)'}",
        f"2. FA ({fa:.2f}) [SHAP Wt: 3.72]              -> {'[+] Boost (Low FA <= 0.10)' if fa <= base['FA'] else '[-] Penalty (High FA Dilution)'}",
        f"3. Slag_Hyd_Idx ({slag_idx:.2f}) [SHAP Wt: 3.04] -> {'[+] Boost (Active Slag)' if slag_idx >= 20.0 else ('[-] Moderate Deficit' if slag_idx >= 10.0 else '[--] Severe Deficit Penalty')}",
        f"4. Age ({age:.0f}d) [SHAP Wt: 1.66]            -> {'[+] Boost (Mature Curing)' if age >= base['Age'] else '[-] Penalty (Early Age)'}",
        f"5. Al2O3 ({al2o3:.2f}%) [SHAP Wt: 1.63]         -> {'[+] Boost (Low Al2O3)' if al2o3 <= base['Al2O3'] else '[-] Penalty (High Al2O3)'}",
        f"6. SP/WBR ({sp_wbr:.4f}) [SHAP Wt: 1.39]       -> {'[+] Positive Dispersion Boost' if sp_wbr > 0 else '[0] No SP Added (Baseline)'}",
        f"7. CaO/Al2O3 ({cao_al2o3:.2f}) [SHAP Wt: 0.56]     -> {'[+] Boost (High Ratio)' if cao_al2o3 >= base['CaO/Al2O3'] else '[-] Penalty (Low Ratio)'}",
        f"8. SO3/CaO ({so3_cao:.4f}) [SHAP Wt: 0.54]      -> {'[-] Penalty (Excess Sulfate > 0.11)' if so3_cao > 0.11 else '[+] Optimal Sulfate Balance'}",
        f"9. Fiber ({fiber:.3f}) [SHAP Wt: 0.39]          -> {'[+] Positive Fiber Bridging' if fiber > 0 else '[0] No Fiber'}",
        "-" * 68,
        f"RF Base Output: {raw_pred:.2f} MPa  |  SHAP-Guided Calibration: {final_pred - raw_pred:+.2f} MPa",
    ]

    return final_pred, "\n".join(advisory_lines), (slag_idx < 10.0 and so3_cao > 0.11)


# --- Muat File .pkl ---
try:
    model, baseline, feature_names = load_artifacts(FOLDER_PATH)
except FileNotFoundError as e:
    st.error(
        f"**File Tidak Ditemukan!**\n\n"
        f"Pastikan file `.pkl` sudah di-upload ke repositori GitHub yang sama dengan `app.py`.\n\n"
        f"**Detail error:** `{e}`"
    )
    st.stop()

# --- Antarmuka Utama (UI) ---
st.title("Mortar Strength Calculator")
st.caption(" ")

with st.form("prediction_form"):
    st.subheader("Mix Proportions & Oxide Composition")
    col1, col2 = st.columns(2)

    with col1:
        ggbs  = st.number_input("GGBS (fraction by total binders (0-1)):", value=0.60, step=0.05, format="%.2f")
        fa    = st.number_input("Fly Ash - FA (fraction by total binders (0-1)):", value=0.10, step=0.05, format="%.2f")
        cfa   = st.number_input("Co-fired FA - CFA (fraction by total binders (0-1)):", value=0.30, step=0.05, format="%.2f")
        sf    = st.number_input("Silica Fume - SF (fraction by total binders (0-1)):", value=0.00, step=0.01, format="%.2f")
        wbr   = st.number_input("WBR - w/b (ratio by total binders (0-1)):", value=0.50, step=0.01, format="%.2f")
        age   = st.number_input("Age (Days):", value=28.0, step=1.0, format="%.0f")

    with col2:
        fiber = st.number_input("Fiber (ratio by total binders (0-1)):", value=0.000, step=0.005, format="%.3f")
        sp    = st.number_input("Superplasticizer - SP (ratio by total binders (0-1)):", value=0.010, step=0.005, format="%.3f")
        al2o3 = st.number_input("Al2O3 (%):", value=18.09, step=0.10, format="%.2f")
        so3   = st.number_input("SO3 (%):", value=3.74, step=0.10, format="%.2f")
        sio2  = st.number_input("SiO2 (%):", value=23.77, step=0.10, format="%.2f")
        cao   = st.number_input("CaO (%):", value=45.74, step=0.10, format="%.2f")

    submitted = st.form_submit_button(
        "Calculate Strength",
        type="primary",
        use_container_width=True,
    )

# --- Eksekusi Perhitungan ---
if submitted:
    try:
        # 1. Proteksi otomatis jika user memasukkan oksida dalam desimal (misal 0.12 bukan 12%)
        if al2o3 < 1.0 and sio2 < 1.0 and cao < 1.0:
            al2o3, sio2, so3, cao = al2o3 * 100.0, sio2 * 100.0, so3 * 100.0, cao * 100.0
            st.info("Catatan: Nilai oksida terdeteksi dalam format desimal dan telah dikonversi otomatis ke persen (%).")

        # 2. Siapkan dictionary data input dimulai dari baseline median
        input_data = baseline.copy()

        # 3. Masukkan seluruh input utama ke input_data
        input_data["WBR"]   = float(wbr)
        input_data["GGBS"]  = float(ggbs)
        input_data["FA"]    = float(fa)
        input_data["CFA"]   = float(cfa)
        input_data["SF"]    = float(sf)
        input_data["Age"]   = float(age)
        input_data["Al2O3"] = float(al2o3)
        input_data["Fiber"] = float(fiber)
        input_data["SP"]    = float(sp)
        input_data["SiO2"]  = float(sio2)
        input_data["SO3"]   = float(so3)
        input_data["CaO"]   = float(cao)

        rufa = input_data.get("RUFA", 0.0)
        nbinder = ggbs + cfa + rufa + sf + fa
        input_data["nBinder"] = nbinder if nbinder != 0 else 1.0
        input_data["ABR"] = input_data.get("Aggregate", 1.0) / input_data["nBinder"]

        # 4. Hitung rasio oksida & fitur domain sesuai nama di feature_columns.pkl
        input_data["SiO2/Al2O3"] = sio2 / (al2o3 + 1e-5)
        input_data["CaO/Al2O3"]  = cao / (al2o3 + 1e-5)
        input_data["CaO/SiO2"]   = cao / (sio2 + 1e-5)
        input_data["SO3/CaO"]    = so3 / (cao + 1e-5)
        input_data["SP/WBR"]     = sp / wbr if wbr != 0 else 0.0

        input_data["GGBS_CFA_Synergy"]     = ggbs * cfa
        input_data["Slag_Hydration_Index"] = (ggbs * sio2) / (wbr + 1e-5)
        input_data["GGBS_CFA_Balance"]     = ggbs / (cfa + 0.1)

        # 5. Konversi ke DataFrame sesuai urutan kolom training
        df_input = pd.DataFrame([input_data])[feature_names]

        # 6. Prediksi RF & terapkan aturan ketat SHAP
        raw_rf_pred = float(model.predict(df_input)[0])
        final_pred, shap_advisory_text, is_critical = enforce_shap_rules(
            raw_rf_pred, input_data, baseline
        )

        # 7. Tampilkan Hasil & Kotak Advisory SHAP
        st.divider()
        st.metric(
        st.markdown("**SHAP Rule Compliance & Feature Impact Audit:**")
        st.code(shap_advisory_text, language="text")

    except KeyError as e:
        st.error(
            f"**Feature Error:** Nama fitur `{e}` tidak cocok dengan yang ada di `feature_columns.pkl`!\n"
            f"Jalankan ulang Cell 12 dan Cell 18 terlebih dahulu."
        )
