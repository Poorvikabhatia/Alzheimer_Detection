import streamlit as st
import numpy as np
import pickle

# Load the trained model
with open('Alzheimer_Detection_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title
st.title("🧠 Alzheimer's Disease Detection")
st.write("Enter the patient details below:")

st.set_page_config(page_title="Alzheimer's Prediction", page_icon="🧠", layout="centered")


with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRVKBS0PBLnm1XxCeszSeZ8Minh4pqN_zlF3w&s", width=100)
    st.title("🧠 Alzheimer's Detector")
    st.markdown("""
    This tool uses a trained machine learning model to detect early signs of Alzheimer's disease based on patient health data.

    **Instructions:**
    - Fill in all the input values
    - Click **Predict Diagnosis**
    - Get instant feedback with confidence score
    """)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://xmple.com/wallpaper/pink-blue-gradient-linear-4500x3000-c2-87cefa-ffb6c1-a-285-f-14.svg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    h1, h2, h3 {{
        color: #2c3e50;
    }}
    .stButton>button {{
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Collect user input
FunctionalAssessment = st.number_input("Functional Assessment (0.0–10.0)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
ADL = st.number_input("Activities of Daily Living (ADL) (0.0–10.0)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
MMSE = st.number_input("MMSE Score (0.0–30.0)", min_value=0.0, max_value=30.0, value=24.0, step=0.1)
MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])
BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])
DietQuality = st.number_input("Diet Quality (0.0–10.0)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
CholesterolTriglycerides = st.number_input("Cholesterol Triglycerides (mg/dL)", min_value=50.0, max_value=400.0, value=150.0, step=0.1)
PhysicalActivity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
CholesterolHDL = st.number_input("Cholesterol HDL (mg/dL)", min_value=20.0, max_value=100.0, value=50.0, step=0.1)
SleepQuality = st.number_input("Sleep Quality (4.0–10.0)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
CholesterolLDL = st.number_input("Cholesterol LDL (mg/dL)", min_value=50.0, max_value=200.0, value=100.0, step=0.1)
AlcoholConsumption = st.number_input("Alcohol Consumption (units/week)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
CholesterolTotal = st.number_input("Cholesterol Total (mg/dL)", min_value=150.0, max_value=300.0, value=200.0, step=0.1)
SystolicBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=90.0, max_value=180.0, value=130.0, step=1.0)
DiastolicBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=60.0, max_value=120.0, value=85.0, step=1.0)
Age = st.number_input("Age", min_value=40.0, max_value=100.0, value=65.0, step=1.0)

# Prediction button
if st.button("Predict Diagnosis"):
    input_data = np.array([[FunctionalAssessment, ADL, MMSE, MemoryComplaints, BehavioralProblems,
                            DietQuality, CholesterolTriglycerides, PhysicalActivity, CholesterolHDL,
                            SleepQuality, CholesterolLDL, AlcoholConsumption, CholesterolTotal,
                            SystolicBP, Age, DiastolicBP]])
    
    prediction = model.predict(input_data)[0]
    diagnosis = "🟥 Alzheimer's Detected (1)" if prediction == 1 else "🟩 No Alzheimer's (0)"

    st.subheader("🩺 Prediction Result:")
    st.success(diagnosis)
