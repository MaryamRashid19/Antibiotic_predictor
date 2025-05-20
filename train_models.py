import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess_data(filepath):
    """Load and preprocess the data"""
    # Load data
    data = pd.read_csv(filepath)
    
    # Handle missing values
    data['allergies'].fillna('None', inplace=True)
    data.dropna(subset=['recommended_antibiotic'], inplace=True)
    
    return data

def preprocess_data(df):
    """Process the data for model training"""
    # Process categorical data
    label_encoders = {}
    categorical_cols = ['gender', 'infection_type', 'allergies', 'severity_level']
    
    df_processed = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Process symptoms
    symptoms_dummies = df_processed['symptoms'].str.get_dummies(sep=', ')
    symptom_columns = symptoms_dummies.columns.tolist()
    df_processed = pd.concat([df_processed, symptoms_dummies], axis=1)
    
    # Prepare final dataset
    df_processed.drop(['symptoms', 'patient_id'], axis=1, inplace=True)
    
    return df_processed, label_encoders, symptom_columns

def train_and_save_model():
    """Main function to train and save the model"""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data('data/data.csv')
    data_processed, label_encoders, symptom_columns = preprocess_data(data)
    
    # Train model
    X = data_processed.drop('recommended_antibiotic', axis=1)
    y = data_processed['recommended_antibiotic']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'models/antibiotic_model.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(symptom_columns, 'models/symptom_columns.pkl')
    
    print("Model training complete and files saved to models/ directory")

if __name__ == "__main__":
    train_and_save_model()