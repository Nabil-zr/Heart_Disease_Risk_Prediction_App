import streamlit as st
import pandas as pd
import pickle
import os
#from sklearn import set_config
#set_config(transform_output='pandas')

# Define the path where app.py is saved, usually /content
base_path = '/content'

# Create .streamlit inside /content
os.makedirs(os.path.join(base_path, '.streamlit'), exist_ok=True)

# Write config.toml inside .streamlit folder
with open(os.path.join(base_path, '.streamlit', 'config.toml'), 'w') as f:
    f.write("""
[theme]
base="dark"
""")

trained_pipe = pickle.load(open('/content/trained_pipe/trained_pipe_LogReg.sav', 'rb'))


st.title("Heart Disease Prediction App")

st.write("""
### Project description
This app predicts the risk of heart disease using a machine learning model with an accuracy of 81%. 
Please note that this is not an official medical tool but a project developed as part of a data science course. 
Always consult a healthcare professional for medical advice.

""")


# Inputs
BMI = st.number_input("BMI (Body Mass Index: weight (kg) / height^2 (m))", min_value=10.0, max_value=60.0, step=0.1)
Smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
AlcoholDrinking = st.selectbox("Do you consume alcohol heavily?", ["Yes", "No"])
Stroke = st.selectbox("Have you had a stroke?", ["Yes", "No"])
PhysicalHealth = st.number_input("Physical Health (days affected in past 30 days)", min_value=0, max_value=30)
MentalHealth = st.number_input("Mental Health (days affected in past 30 days)", min_value=0, max_value=30)
DiffWalking = st.selectbox("Do you have difficulty walking?", ["Yes", "No"])
Sex = st.selectbox("Sex", ["Male", "Female"])
AgeCategory = st.selectbox("Age Category", [
    '18-24', '25-29', '30-34', '35-39', '40-44',
    '45-49', '50-54', '55-59', '60-64', '65-69',
    '70-74', '75-79', '80 or older'
])
Race = st.selectbox("Race", [
    "White", "Black", "Asian", "American Indian/Alaskan Native",
    "Other", "Hispanic"
])
Diabetic = st.selectbox("Are you diabetic?", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
PhysicalActivity = st.selectbox("Do you engage in physical activity?", ["Yes", "No"])
GenHealth = st.selectbox("General Health", ['Very good', 'Fair', 'Good', 'Poor', 'Excellent'])

SleepTime = st.number_input("Average Sleep Time (hours/day)", min_value=0.0, max_value=24.0, step=0.5)
Asthma = st.selectbox("Do you have asthma?", ["Yes", "No"])
KidneyDisease = st.selectbox("Do you have kidney disease?", ["Yes", "No"])
SkinCancer = st.selectbox("Do you have skin cancer?", ["Yes", "No"])


# ----- Make Prediction -----

if st.button("Predict"):
    input_data = pd.DataFrame({
        'BMI': [BMI],
        'Smoking': [Smoking],
        'AlcoholDrinking': [AlcoholDrinking],
        'Stroke': [Stroke],
        'PhysicalHealth': [PhysicalHealth],
        'MentalHealth': [MentalHealth],
        'DiffWalking': [DiffWalking],
        'Sex': [Sex],
        'AgeCategory': [AgeCategory],
        'Race': [Race],
        'Diabetic': [Diabetic],
        'PhysicalActivity': [PhysicalActivity],
        'GenHealth': [GenHealth],
        'SleepTime': [SleepTime],
        'Asthma': [Asthma],
        'KidneyDisease': [KidneyDisease],
        'SkinCancer': [SkinCancer]
    })

    st.write("🧪 Input Data Preview:")
    st.dataframe(input_data)

  # Predict using the pipeline

    prediction = trained_pipe.predict(input_data)




    prob = trained_pipe.predict_proba(input_data)[0][1]  # Probability of heart disease

    if prob > 0.70:
        st.error(f"⚠️ Very High Risk of heart disease! (Probability: {prob:.2%})")
    elif 0.45 < prob <= 0.70:
        st.warning(f"⚠️ Moderate Risk of heart disease. (Probability: {prob:.2%})")
    elif 0 < prob <= 0.45:
        st.info(f"ℹ️ Low Risk detected. (Probability: {prob:.2%})")
    else:
        st.success(f"✅ No risk detected. (Probability: {prob:.2%})")


    #if prediction[0] == 1:
      #st.error(f"⚠️ High risk of heart disease! (Probability: {prob:.2%})")
    #else:
      #st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2%})")

    #prob = trained_pipe.predict_proba(input_data)[0][1]  # Probability of heart disease

    #if prediction[0] == 1:
      #st.error(f"⚠️ High risk of heart disease! (Probability: {prob:.2%})")
    #else:
      #st.success(f"✅ Low risk of heart disease. (Probability: {prob:.2%})")



