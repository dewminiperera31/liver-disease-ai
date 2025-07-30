import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

st.title("🧬 Liver Disease Outcome Prediction")

# === Load model ===
with open(r"C:\Users\Avishka\Desktop\AI for Liver Disease Outcome Prediction\model\model.pkl", "rb") as f:
    model = pickle.load(f)


# === Label encoders (same as training) ===
le_drug = LabelEncoder()
le_drug.fit(['D-penicillamine', 'Placebo'])

le_sex = LabelEncoder()
le_sex.fit(['Male', 'Female'])

le_age_bin = LabelEncoder()
le_age_bin.fit(['<35', '35-50', '>50'])

le_stage_grouped = LabelEncoder()
le_stage_grouped.fit(['low', 'mid', 'high'])

# === Input widgets ===
id = st.number_input("Patient ID", min_value=1, value=1)
N_Days = st.slider("Days in Study", 1, 500, 150)
Drug = st.selectbox("Drug", ['D-penicillamine', 'Placebo'])
Age = st.slider("Age", 18, 100, 45)
Sex = st.radio("Sex", ['Male', 'Female'])
Ascites = st.radio("Ascites", [0, 1])
Hepatomegaly = st.radio("Hepatomegaly", [0, 1])
Spiders = st.radio("Spiders", [0, 1])
Edema = st.radio("Edema", [0, 1])
Bilirubin = st.number_input("Bilirubin", min_value=0.1, value=1.8)
Cholesterol = st.number_input("Cholesterol", min_value=50.0, value=200.0)
Albumin = st.number_input("Albumin", min_value=1.0, value=3.5)
Copper = st.number_input("Copper", min_value=0.0, value=80.0)
Alk_Phos = st.number_input("Alk_Phos", min_value=20.0, value=120.0)
SGOT = st.number_input("SGOT", min_value=10.0, value=90.0)
Tryglicerides = st.number_input("Tryglicerides", min_value=10.0, value=150.0)
Platelets = st.number_input("Platelets", min_value=10000.0, value=250000.0)
Prothrombin = st.number_input("Prothrombin", min_value=5.0, value=10.0)
Stage = st.slider("Stage", 1, 4, 2)

# Derived features
Age_bin = '<35' if Age < 35 else ('35-50' if Age <= 50 else '>50')
Stage_grouped = 'low' if Stage == 1 else ('mid' if Stage == 2 else 'high')

# On Predict button
if st.button("Predict Outcome"):
    # Encode categorical
    Drug_enc = le_drug.transform([Drug])[0]
    Sex_enc = le_sex.transform([Sex])[0]
    Age_bin_enc = le_age_bin.transform([Age_bin])[0]
    Stage_grouped_enc = le_stage_grouped.transform([Stage_grouped])[0]

    # Engineer features
    Bili_Alb = Bilirubin / Albumin if Albumin != 0 else 0
    Age_Stage = Age * Stage
    Prothrombin_Platelets_Ratio = Prothrombin / Platelets if Platelets != 0 else 0
    Bilirubin_log = np.log(Bilirubin)
    Cholesterol_log = np.log(Cholesterol)
    SGOT_log = np.log(SGOT)
    Bilirubin_bin_1 = 1 if Bilirubin < 1.2 else 0
    Bilirubin_bin_2 = 1 if 1.2 <= Bilirubin < 2.0 else 0

    # Build input DataFrame
    input_dict = {
        'id': id, 'N_Days': N_Days, 'Drug': Drug_enc, 'Age': Age, 'Sex': Sex_enc,
        'Ascites': Ascites, 'Hepatomegaly': Hepatomegaly, 'Spiders': Spiders, 'Edema': Edema,
        'Bilirubin': Bilirubin, 'Cholesterol': Cholesterol, 'Albumin': Albumin, 'Copper': Copper,
        'Alk_Phos': Alk_Phos, 'SGOT': SGOT, 'Tryglicerides': Tryglicerides,
        'Platelets': Platelets, 'Prothrombin': Prothrombin, 'Stage': Stage,
        'Bili_Alb': Bili_Alb, 'Age_Stage': Age_Stage, 'Prothrombin_Platelets_Ratio': Prothrombin_Platelets_Ratio,
        'Age_bin': Age_bin_enc, 'Stage_grouped': Stage_grouped_enc,
        'Bilirubin_log': Bilirubin_log, 'Cholesterol_log': Cholesterol_log,
        'SGOT_log': SGOT_log, 'Bilirubin_bin_1': Bilirubin_bin_1, 'Bilirubin_bin_2': Bilirubin_bin_2
    }

    feature_order = list(input_dict.keys())
    input_df = pd.DataFrame([input_dict], columns=feature_order)

    # Predict
    prediction = model.predict(input_df)[0]
    outcome = 'Lived' if prediction == 'L' else 'Died'

    st.success(f"🩺 Predicted Outcome: **{outcome}**")
