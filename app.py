import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Title */
h1 {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Labels */
label {
    font-weight: 500 !important;
    color: #334155 !important;
}

/* Button */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: all 0.2s ease;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

/* Result */
.result {
    background-color: #ecfdf5;
    border: 1px solid #bbf7d0;
    padding: 18px;
    border-radius: 10px;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    color: #065f46;
    margin-top: 20px;
}

/* Spacing */
.stSlider, .stSelectbox {
    margin-bottom: 18px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("salary_model.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("AI Salary Predictor")
st.markdown(
    "<div class='subtitle'>Accurate salary estimates powered by machine learning</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    experience = st.slider("Experience (Years)", 0, 20, 2)
    employment = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Freelance"]
    )

with col2:
    company_size = st.selectbox(
        "Company Size",
        ["Small", "Medium", "Large"]
    )
    remote_ratio = st.slider("Remote Work (%)", 0, 100, 50)

location = st.selectbox(
    "Company Location",
    ["USA", "Europe", "Asia", "Africa", "Other"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ENCODING ----------------
employment_map = {
    "Full-time": 0,
    "Part-time": 1,
    "Contract": 2,
    "Freelance": 3
}

company_size_map = {
    "Small": 0,
    "Medium": 1,
    "Large": 2
}

location_map = {
    "USA": 0,
    "Europe": 1,
    "Asia": 2,
    "Africa": 3,
    "Other": 4
}

# ---------------- PREDICTION ----------------
if st.button("Predict Salary"):
    features = np.array([[
        experience,
        employment_map[employment],
        company_size_map[company_size],
        remote_ratio,
        location_map[location]
    ]])

    prediction = model.predict(features)

    st.markdown(
        f"<div class='result'>Estimated Salary: ${prediction[0]:,.2f}</div>",
        unsafe_allow_html=True
    )