# import streamlit as st
# import pickle
# import pandas as pd

# # Load trained models
# rf = pickle.load(open("model/rf.pkl", "rb"))
# svc = pickle.load(open("model/svc.pkl", "rb"))

# # Load dataset to get symptom names
# dataset = pd.read_csv("Training.csv")

# # Get symptom columns
# symptoms = dataset.drop("prognosis", axis=1).columns.tolist()

# st.title("🩺 Medicine Recommendation System")

# st.write("Select your symptoms to predict the disease.")

# # Select symptoms
# selected_symptoms = st.multiselect(
#     "Select your symptoms:",
#     symptoms
# )

# # Predict button
# if st.button("Predict Disease"):

#     if len(selected_symptoms) == 0:
#         st.warning("Please select at least one symptom.")

#     else:
#         # Create input with all symptoms = 0
#         input_data = pd.DataFrame(
#             0,
#             index=[0],
#             columns=symptoms
#         )

#         # Set selected symptoms to 1
#         for symptom in selected_symptoms:
#             input_data[symptom] = 1

#         # Prediction
#         prediction = svc.predict(input_data)

#         st.success(f"Predicted Disease: {prediction[0]}")
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained models
with open("model/rf.pkl", "rb") as f:
    rf = pickle.load(f)

with open("model/svc.pkl", "rb") as f:
    svc = pickle.load(f)

# Load the dataset
dataset = pd.read_csv("Training.csv")

# Get the exact features used when the model was trained
model_features = list(svc.feature_names_in_)

# Show only actual symptoms on the website
# Hide the unwanted "Unnamed: 133" column
symptoms = [
    col for col in model_features
    if not col.startswith("Unnamed")
]


# Home page
@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=symptoms
    )


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    selected_symptoms = data.get("symptoms", [])

    # Check whether symptoms were selected
    if not selected_symptoms:
        return jsonify({
            "error": "Please select at least one symptom."
        }), 400

    # Create input with EXACTLY the same features
    # that were used during model training
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    # Set selected symptoms to 1
    for symptom in selected_symptoms:

        if symptom in input_data.columns:
            input_data[symptom] = 1

    # Predict disease using SVC
    prediction = svc.predict(input_data)

    # Convert NumPy value to normal Python string
    # so Flask can return it as JSON
    return jsonify({
        "disease": str(prediction[0])
    })


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)