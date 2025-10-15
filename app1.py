
import streamlit as st
import pickle
import numpy as np

# Load model
model_path = r"C:\Users\K. BHAVANI\Assignments-DS\titanic_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction App")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])
age = st.slider("Age", 1, 100, 25)
sibsp = st.slider("Number of Siblings/Spouses Aboard", 0, 8)
parch = st.slider("Number of Parents/Children Aboard", 0, 6)
fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Encoding (use the same encoders or map as you did in training)
sex_encoded = 1 if sex == "male" else 0
embarked_mapping = {"S": 2, "C": 0, "Q": 1}  # example based on your notebook label encoder
embarked_encoded = embarked_mapping[embarked]
title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Other": 4}
title_encoded = title_mapping[title]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded, title_encoded]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The passenger would likely SURVIVE!")
    else:
        st.error("The passenger would NOT survive.")

