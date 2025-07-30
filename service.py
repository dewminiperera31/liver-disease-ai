from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
CORS(app)

# Load model from model folder - tries model.pkl first, then final_model.pkl
MODEL_PATHS = [
    os.path.join("model", "model.pkl"),
    os.path.join("model", "final_model.pkl")
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Loaded model from {path}")
        break

if model is None:
    raise FileNotFoundError("No model file found in the 'model' folder.")

# Setup label encoders (must match your training)
le_drug = LabelEncoder()
le_drug.fit(['D-penicillamine', 'Placebo'])

le_sex = LabelEncoder()
le_sex.fit(['Male', 'Female'])

le_age_bin = LabelEncoder()
le_age_bin.fit(['<35', '35-50', '>50'])

le_stage_grouped = LabelEncoder()
le_stage_grouped.fit(['low', 'mid', 'high'])

@app.route("/")
def home():
    return jsonify({"message": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Extract input fields (use .get with defaults where appropriate)
        id = data.get("id", 1)
        N_Days = data["N_Days"]
        Drug = data["Drug"]
        Age = data["Age"]
        Sex = data["Sex"]
        Ascites = data["Ascites"]
        Hepatomegaly = data["Hepatomegaly"]
        Spiders = data["Spiders"]
        Edema = data["Edema"]
        Bilirubin = data["Bilirubin"]
        Cholesterol = data["Cholesterol"]
        Albumin = data["Albumin"]
        Copper = data["Copper"]
        Alk_Phos = data["Alk_Phos"]
        SGOT = data["SGOT"]
        Tryglicerides = data["Tryglicerides"]
        Platelets = data["Platelets"]
        Prothrombin = data["Prothrombin"]
        Stage = data["Stage"]

        # Derived categorical features
        Age_bin = '<35' if Age < 35 else ('35-50' if Age <= 50 else '>50')
        Stage_grouped = 'low' if Stage == 1 else ('mid' if Stage == 2 else 'high')

        # Encode categorical variables
        Drug_enc = le_drug.transform([Drug])[0]
        Sex_enc = le_sex.transform([Sex])[0]
        Age_bin_enc = le_age_bin.transform([Age_bin])[0]
        Stage_grouped_enc = le_stage_grouped.transform([Stage_grouped])[0]

        # Engineered numeric features
        Bili_Alb = Bilirubin / Albumin if Albumin != 0 else 0
        Age_Stage = Age * Stage
        Prothrombin_Platelets_Ratio = Prothrombin / Platelets if Platelets != 0 else 0
        Bilirubin_log = np.log(Bilirubin) if Bilirubin > 0 else 0
        Cholesterol_log = np.log(Cholesterol) if Cholesterol > 0 else 0
        SGOT_log = np.log(SGOT) if SGOT > 0 else 0
        Bilirubin_bin_1 = 1 if Bilirubin < 1.2 else 0
        Bilirubin_bin_2 = 1 if 1.2 <= Bilirubin < 2.0 else 0

        # Build input DataFrame with exact feature columns expected by your model
        input_dict = {
            'id': id,
            'N_Days': N_Days,
            'Drug': Drug_enc,
            'Age': Age,
            'Sex': Sex_enc,
            'Ascites': Ascites,
            'Hepatomegaly': Hepatomegaly,
            'Spiders': Spiders,
            'Edema': Edema,
            'Bilirubin': Bilirubin,
            'Cholesterol': Cholesterol,
            'Albumin': Albumin,
            'Copper': Copper,
            'Alk_Phos': Alk_Phos,
            'SGOT': SGOT,
            'Tryglicerides': Tryglicerides,
            'Platelets': Platelets,
            'Prothrombin': Prothrombin,
            'Stage': Stage,
            'Bili_Alb': Bili_Alb,
            'Age_Stage': Age_Stage,
            'Prothrombin_Platelets_Ratio': Prothrombin_Platelets_Ratio,
            'Age_bin': Age_bin_enc,
            'Stage_grouped': Stage_grouped_enc,
            'Bilirubin_log': Bilirubin_log,
            'Cholesterol_log': Cholesterol_log,
            'SGOT_log': SGOT_log,
            'Bilirubin_bin_1': Bilirubin_bin_1,
            'Bilirubin_bin_2': Bilirubin_bin_2
        }

        df = pd.DataFrame([input_dict])

        prediction = model.predict(df)[0]
        outcome = "Lived" if prediction == "L" else "Died"

        return jsonify({"prediction": outcome})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
