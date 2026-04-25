"""
Unified Data Preprocessing Module
Provides consistent data loading for all models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Feature definitions
NUMERICAL_FEATURES = [
    'Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count',
    'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils', 'Thrombocyte_Count',
    'CRP', 'Neutrophil_Percentage'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite',
    'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 'Severity',
    'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness',
    'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine'
]

# Laboratory features for missing indicators
LABORATORY_FEATURES = [
    'WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils',
    'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage'
]

# Missing indicators for laboratory features
LAB_MISSING_INDICATORS = [f"{lab}_missing" for lab in LABORATORY_FEATURES]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + LAB_MISSING_INDICATORS

def create_synthetic_dataset(n_samples=1000):
    """Create synthetic dataset for pediatric appendicitis prediction"""
    np.random.seed(42)
    
    data = {}
    
    # Numerical features
    data['Age'] = np.random.normal(10, 3, n_samples).clip(1, 18)
    data['Weight'] = np.random.normal(35, 15, n_samples).clip(10, 100)
    data['Height'] = np.random.normal(140, 20, n_samples).clip(80, 200)
    data['BMI'] = data['Weight'] / (data['Height']/100)**2
    data['Body_Temperature'] = np.random.normal(37.5, 0.8, n_samples).clip(36, 40)
    data['WBC_Count'] = np.random.normal(12, 5, n_samples).clip(2, 30)
    data['RBC_Count'] = np.random.normal(4.5, 0.5, n_samples).clip(3, 6)
    data['Hemoglobin'] = np.random.normal(12, 2, n_samples).clip(8, 16)
    data['RDW'] = np.random.normal(13, 2, n_samples).clip(10, 20)
    data['Segmented_Neutrophils'] = np.random.normal(70, 15, n_samples).clip(20, 95)
    data['Thrombocyte_Count'] = np.random.normal(250, 100, n_samples).clip(50, 600)
    data['CRP'] = np.random.exponential(20, n_samples).clip(0, 200)
    data['Neutrophil_Percentage'] = np.random.normal(75, 10, n_samples).clip(30, 95)
    
    # Categorical features
    data['Sex'] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    data['Lower_Right_Abd_Pain'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    data['Migratory_Pain'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    data['Loss_of_Appetite'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    data['Nausea'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    data['Coughing_Pain'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['Dysuria'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    data['Stool'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    data['Peritonitis'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['Severity'] = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
    data['Contralateral_Rebound_Tenderness'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['Ipsilateral_Rebound_Tenderness'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    data['Psoas_Sign'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['Neutrophilia'] = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    data['Ketones_in_Urine'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    data['RBC_in_Urine'] = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    data['WBC_in_Urine'] = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    
    # Create diagnosis (target) with some correlation to features
    # Higher probability of appendicitis with certain symptoms
    risk_score = (
        0.3 * data['Lower_Right_Abd_Pain'] +
        0.2 * data['Migratory_Pain'] +
        0.15 * data['Loss_of_Appetite'] +
        0.1 * data['Nausea'] +
        0.2 * data['Peritonitis'] +
        0.15 * data['Ipsilateral_Rebound_Tenderness'] +
        0.1 * (data['WBC_Count'] > 15) +
        0.1 * (data['CRP'] > 50)
    )
    
    # Add noise and convert to binary
    risk_score += np.random.normal(0, 0.3, n_samples)
    probability = 1 / (1 + np.exp(-risk_score))
    data['Diagnosis'] = (probability > 0.5).astype(int)
    
    # Add missing indicators for laboratory features (30% missing rate)
    for lab in LABORATORY_FEATURES:
        missing_mask = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% missing
        data[f"{lab}_missing"] = missing_mask
        
        # For missing values, set to NaN (will be handled by imputation)
        missing_indices = np.where(missing_mask == 1)[0]
        if len(missing_indices) > 0:
            data[lab][missing_indices] = np.nan
    
    return pd.DataFrame(data)

def prepare_unified_data(data_type='CSV'):
    """Prepare unified data for model training"""
    
    # Create synthetic dataset
    df = create_synthetic_dataset()
    
    # Separate features and target
    X = df[ALL_FEATURES].copy()
    y = df['Diagnosis'].copy()
    
    # Handle missing values properly for lab features
    # Impute actual lab values when missing, but keep missing indicators
    numerical_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    # Impute numerical features (including labs) - missing indicators remain unchanged
    X[NUMERICAL_FEATURES] = numerical_imputer.fit_transform(X[NUMERICAL_FEATURES])
    X[CATEGORICAL_FEATURES] = categorical_imputer.fit_transform(X[CATEGORICAL_FEATURES])
    
    # Missing indicators should remain as 0/1 (no imputation)
    # They are already created in the synthetic dataset
    
    # Scale numerical features
    numerical_scaler = StandardScaler()
    X[NUMERICAL_FEATURES] = numerical_scaler.fit_transform(X[NUMERICAL_FEATURES])
    
    # Encode categorical features (already 0/1, but keep for consistency)
    categorical_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        categorical_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (
        X_train, X_test, y_train, y_test,
        ALL_FEATURES, categorical_encoders, numerical_scaler
    )

if __name__ == "__main__":
    # Test the function
    print("Testing unified data preprocessing...")
    X_train, X_test, y_train, y_test, features, encoders, scaler = prepare_unified_data()
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Features: {len(features)}")
    print(f"Diagnosis distribution: {y_train.value_counts().to_dict()}")
