import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Setup Halaman
st.set_page_config(page_title="Cementless Strength Calculator", layout="centered")

st.title("Cementless Strength Predictor")
st.write(
    "Please input parameter (in ratio 0.0 – 1.0)"
)


# Load model dan metadata dengan cache agar cepat
@st.cache_resource
def load_assets():
  model = joblib.load("best_rf_model.pkl")
  baseline = joblib.load("baseline_medians.pkl")
  features = joblib.load("feature_columns.pkl")
  return model, baseline, features


model, baseline_data, feature_names = load_assets()

# Form Input Pengguna
with st.form("input_form"):
  st.subheader("1. Materials proportion & Age")
  col1, col2 = st.columns(2)

  with col1:
    wbr = st.number_input(
        "Water-to-Binder Ratio (WBR)",
        min_value=0.01,
        max_value=1.0,
        value=0.40,
        step=0.01,
    )
    fa = st.number_input(
        "Fly Ash Ratio (FA)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )
    ggbs = st.number_input(
        "GGBS Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.05,
    )

  with col2:
    age = st.number_input(
        "Age (days)",
        min_value=1,
        max_value=365,
        value=28,
        step=1,
    )
    fiber = st.number_input(
        "Fiber by binders",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.005,
        format="%.3f",
    )
    sp = st.number_input(
        "Superplasticizer (SP) by binders",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
        step=0.005,
        format="%.3f",
    )

  st.subheader("2. Chemichal Ratio")
  col3, col4 = st.columns(2)

  with col3:
    al2o3 = st.number_input(
        "Al2O3",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
    )
    sio2 = st.number_input(
        "SiO2",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
    )

  with col4:
    so3 = st.number_input(
        "SO3",
        min_value=0.0,
        max_value=1.0,
        value=0.03,
        step=0.005,
        format="%.3f",
    )
    cao = st.number_input(
        "CaO",
        min_value=0.01,
        max_value=1.0,
        value=0.10,
        step=0.01,
    )

  # Tombol Submit
  submitted = st.form_submit_button("Compressive strength (Predict)")

# Proses Prediksi
if submitted:
  # 1. Salin baseline 12 fitur minor otomatis
  input_dict = baseline_data.copy()

  # 2. Perbarui dengan nilai input user
  input_dict["WBR"] = wbr
  input_dict["FA"] = fa
  input_dict["Age"] = age
  input_dict["Al2O3"] = al2o3
  input_dict["GGBS"] = ggbs
  input_dict["Fiber"] = fiber

  # 3. Hitung rasio turunan di belakang layar
  input_dict["SP_div_WBR"] = sp / wbr if wbr != 0 else 0
  input_dict["SO3_div_CaO"] = so3 / cao if cao != 0 else 0
  input_dict["SiO2_div_Al2O3"] = sio2 / al2o3 if al2o3 != 0 else 0

  # 4. Susun DataFrame sesuai urutan fitur model pelatihan
  df_input = pd.DataFrame([input_dict])[feature_names]

  # 5. Prediksi nilai
  prediction = model.predict(df_input)[0]

  # Tampilkan Output
  st.success("### Prediction results:")
  st.metric(label="Estimated Compressive Strength", value=f"{prediction:.2f} MPa") 
