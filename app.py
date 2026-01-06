import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- STEP 1: MUST BE THE FIRST ST COMMAND ---
st.set_page_config(
    page_title="Doctor's Heart Prediction Portal", 
    layout="wide",
    page_icon="🩺"
)

# --- STEP 2: LOAD MODEL & ASSETS ---
@st.cache_resource
def load_model():
    try:
        with open('heart_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        return model, encoder
    except FileNotFoundError:
        return None, None

model, encoder = load_model()

# --- STEP 3: APP INTERFACE ---
st.title("🩺 Heart Disease Prediction System")
st.info("Clinical Decision Support Tool for Medical Professionals")

if model is None:
    st.error("⚠️ Model files ('heart_model.pkl' or 'encoder.pkl') not found. Please run your training script first.")
    st.stop()

# Layout using Tabs for better organization
tab1, tab2 = st.tabs(["Single Patient Check", "Batch Analysis (CSV)"])

with tab1:
    st.subheader("Individual Patient Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 1, 110, 50)
        sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], help="1: Typical Angina, 2: Atypical, 3: Non-anginal, 4: Asymptomatic")
        bp = st.number_input("Resting Blood Pressure ($mm Hg$)", 80, 200, 120)
    
    with col2:
        chol = st.number_input("Serum Cholesterol ($mg/dl$)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 $mg/dl$", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
        ekg = st.selectbox("Resting EKG Results", options=[0, 1, 2])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        
    with col3:
        angina = st.selectbox("Exercise Induced Angina", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.selectbox("Slope of ST Segment", options=[1, 2, 3])
        vessels = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
        thallium = st.selectbox("Thallium", options=[3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable"}[x])

    # Prediction Logic
    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'Chest pain type': [cp], 'BP': [bp],
        'Cholesterol': [chol], 'FBS over 120': [fbs], 'EKG results': [ekg],
        'Max HR': [max_hr], 'Exercise angina': [angina], 'ST depression': [oldpeak],
        'Slope of ST': [slope], 'Number of vessels fluro': [vessels], 'Thallium': [thallium]
    })

    if st.button("Generate Diagnostic Report"):
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)
        result = encoder.inverse_transform(prediction)[0]
        
        st.markdown("---")
        if result == 'Presence':
            st.error(f"### High Risk: Heart Disease Detected")
            st.metric("Confidence Score", f"{prob[0][1]*100:.1f}%")
        else:
            st.success(f"### Low Risk: No Heart Disease Detected")
            st.metric("Confidence Score", f"{prob[0][0]*100:.1f}%")

with tab2:
    st.subheader("Process Clinical Records")
    uploaded_file = st.file_uploader("Upload CSV for automated screening", type=["csv"])
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        # Ensure columns match training data
        try:
            # We assume the CSV has the same features except the target 'Heart Disease'
            features = batch_df.copy()
            if 'Heart Disease' in features.columns:
                features = features.drop(columns=['Heart Disease'])
                
            batch_preds = model.predict(features)
            batch_df['Prediction'] = encoder.inverse_transform(batch_preds)
            
            st.dataframe(batch_df)
            st.download_button("Download Predictions", batch_df.to_csv(index=False), "heart_predictions.csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("v1.0.0 | System Operational")
st.sidebar.warning("Note: This AI tool provides predictions based on statistical data. Please correlate with clinical findings.")