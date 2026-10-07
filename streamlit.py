import streamlit as st
import pickle
import pandas as pd

# --------------------------------
# Load trained models
# --------------------------------

with open("model/rf.pkl", "rb") as f:
    rf = pickle.load(f)

with open("model/svc.pkl", "rb") as f:
    svc = pickle.load(f)


# --------------------------------
# Load dataset
# --------------------------------

dataset = pd.read_csv("Training.csv")


# --------------------------------
# Get model features
# --------------------------------

model_features = list(svc.feature_names_in_)

# Show only actual symptoms
symptoms = [
    col for col in model_features
    if not col.startswith("Unnamed")
]


# --------------------------------
# Disease names
# --------------------------------

disease_names = [
    "Acne",
    "Allergy",
    "Arthritis",
    "Bronchial Asthma",
    "Cervical Spondylosis",
    "Chicken Pox",
    "Chronic Cholestasis",
    "Common Cold",
    "Dengue",
    "Diabetes",
    "Dimorphic Hemorrhoids",
    "Drug Reaction",
    "Fungal Infection",
    "GERD",
    "Gastroenteritis",
    "Heart Attack",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Hepatitis F",
    "Hepatitis G",
    "Hypertension",
    "Hyperthyroidism",
    "Hypoglycemia",
    "Hypothyroidism",
    "Impetigo",
    "Jaundice",
    "Malaria",
    "Migraine",
    "Osteoarthritis",
    "Paralysis (Brain Hemorrhage)",
    "Peptic Ulcer Disease",
    "Pneumonia",
    "Psoriasis",
    "Tuberculosis",
    "Typhoid",
    "Urinary Tract Infection",
    "Varicose Veins"
]


# --------------------------------
# Streamlit UI
# --------------------------------

st.title("🩺 Medicine Recommendation System")

st.write(
    "Select your symptoms to predict the disease."
)


# --------------------------------
# Select symptoms
# --------------------------------

selected_symptoms = st.multiselect(
    "Select your symptoms:",
    symptoms
)


# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:

        st.warning(
            "Please select at least one symptom."
        )

    else:

        # Create input with all features = 0
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=model_features
        )

        # Set selected symptoms to 1
        for symptom in selected_symptoms:

            if symptom in input_data.columns:
                input_data[symptom] = 1


        # Predict
        prediction = svc.predict(input_data)

        # Convert prediction to integer
        disease_number = int(prediction[0])


        # Convert number to disease name
        if 0 <= disease_number < len(disease_names):

            disease = disease_names[disease_number]

            st.success(
                f"Predicted Disease: {disease}"
            )

        else:

            st.error(
                f"Unknown disease code: {disease_number}"
            )