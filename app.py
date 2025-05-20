import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_models():
    """Load the trained model and encoders"""
    model = joblib.load('models/antibiotic_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    symptom_columns = joblib.load('models/symptom_columns.pkl')
    return model, label_encoders, symptom_columns

def main():
    # Load models
    try:
        model, label_encoders, symptom_columns = load_models()
    except FileNotFoundError:
        st.error("Model files not found. Please run train_model.py first.")
        return
    
    # Configure page
    st.set_page_config(
        page_title="Antibiotic Recommender",
        page_icon="💊",
        layout="wide"
    )
    
    st.title("💊 Antibiotic Recommendation System")
    st.write("This system recommends antibiotics based on patient characteristics and symptoms.")
    
    # Sidebar for input
    with st.sidebar:
        st.header("Patient Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=70.0)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        infection_type = st.selectbox("Infection Type", ["Bacterial", "Fungal", "Viral"])
        
        st.subheader("Symptoms")
        symptoms = st.multiselect(
            "Select symptoms",
            options=symptom_columns,
            default=[]
        )
        
        st.subheader("Additional Information")
        allergies = st.selectbox("Allergies", ["None", "Sulfa", "Penicillin", "Aspirin"])
        severity = st.selectbox("Severity Level", ["Mild", "Moderate", "Severe"])
        
        if st.button("Recommend Antibiotic", type="primary"):
            # Prepare input data
            try:
                input_data = {
                    'age': age,
                    'weight_kg': weight,
                    'gender': label_encoders['gender'].transform([gender])[0],
                    'infection_type': label_encoders['infection_type'].transform([infection_type])[0],
                    'allergies': label_encoders['allergies'].transform([allergies])[0],
                    'severity_level': label_encoders['severity_level'].transform([severity])[0],
                    **{symptom: 1 for symptom in symptoms}
                }
                
                # Ensure all columns are present
                for symptom in symptom_columns:
                    if symptom not in input_data:
                        input_data[symptom] = 0
                
                # Convert to DataFrame with correct column order
                input_df = pd.DataFrame([input_data])[model.feature_names_in_]
                
                # Make prediction
                recommendation = model.predict(input_df)[0]
                
                # Store results in session state
                st.session_state.recommendation = recommendation
                st.session_state.patient_data = {
                    'age': age,
                    'weight': weight,
                    'gender': gender,
                    'infection_type': infection_type,
                    'symptoms': symptoms,
                    'allergies': allergies,
                    'severity': severity
                }
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    # Display results if available
    if 'recommendation' in st.session_state:
        st.header("Recommendation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Details")
            st.json({
                "Age": st.session_state.patient_data['age'],
                "Weight": f"{st.session_state.patient_data['weight']} kg",
                "Gender": st.session_state.patient_data['gender'],
                "Infection Type": st.session_state.patient_data['infection_type']
            })
        
        with col2:
            st.subheader("Clinical Information")
            st.json({
                "Symptoms": ", ".join(st.session_state.patient_data['symptoms']) or "None",
                "Allergies": st.session_state.patient_data['allergies'],
                "Severity": st.session_state.patient_data['severity']
            })
        
        st.divider()
        
        if st.session_state.recommendation == "None":
            st.success("## Recommendation: Supportive Care (no antibiotic needed)")
        else:
            st.success(f"## Recommended Antibiotic: {st.session_state.recommendation}")

if __name__ == "__main__":
    main()