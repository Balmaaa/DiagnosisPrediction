"""
Unified Data Preprocessing Module
All models use the same data source and preprocessing for fair comparison
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# Define the 30 features used by all models
NUMERICAL_FEATURES = [
    'Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 
    'WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 
    'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
    'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 
    'Severity', 'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 
    'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def load_original_dataset():
    """Load the original dataset from Excel file"""
    
    script_dir = Path(__file__).parent
    data_path = script_dir / "Dataset 1" / "app_data.xlsx"
    
    df = pd.read_excel(data_path)
    print(f"[STEP 5] Loaded original dataset: {df.shape}")
    
    return df

def prepare_unified_data(dataset_name='CSV'):
    """
    Prepare unified data for ALL models
    Returns: X_train, X_test, y_train, y_test, feature_names, encoders, scalers
    """
    
    print(f"\n{'='*60}")
    print(f"[STEP 5] PREPARING UNIFIED DATA FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load original dataset
    df = load_original_dataset()
    
    # Extract target
    if 'Management' in df.columns:
        y = df['Management'].values
    elif 'Diagnosis' in df.columns:
        y = df['Diagnosis'].values
    else:
        raise ValueError("No target column found")
    
    # Convert to binary: surgical cases (1) vs conservative (0)
    y_binary = np.array([1 if str(x).lower() in ['primary surgical', 'secondary surgical', 'simultaneous appendectomy'] else 0 for x in y])
    
    # Select only the 30 features
    X = df[ALL_FEATURES].copy()
    
    print(f"[STEP 5] Selected {len(ALL_FEATURES)} features: {ALL_FEATURES}")
    print(f"[STEP 5] Feature count: {X.shape[1]}")
    
    # Remove rows with missing target
    valid_mask = ~pd.isna(y_binary)
    X = X[valid_mask].reset_index(drop=True)
    y_binary = y_binary[valid_mask]
    
    print(f"[STEP 5] After removing missing targets: {X.shape[0]} samples")
    
    # Handle missing values in features
    for feat in NUMERICAL_FEATURES:
        if feat in X.columns:
            if X[feat].isna().any():
                mean_val = X[feat].mean()
                X[feat].fillna(mean_val, inplace=True)
                print(f"[STEP 5] Filled {X[feat].isna().sum()} NaN in {feat} with mean {mean_val:.2f}")
    
    for feat in CATEGORICAL_FEATURES:
        if feat in X.columns:
            if X[feat].isna().any():
                mode_val = X[feat].mode()[0]
                X[feat].fillna(mode_val, inplace=True)
                print(f"[STEP 5] Filled {X[feat].isna().sum()} NaN in {feat} with mode")
    
    # Encode categorical features with LabelEncoder
    categorical_encoders = {}
    for feat in CATEGORICAL_FEATURES:
        if feat in X.columns:
            le = LabelEncoder()
            X[feat] = le.fit_transform(X[feat].astype(str))
            categorical_encoders[feat] = le
            print(f"[STEP 5] Encoded {feat}: {len(le.classes_)} categories")
    
    # Scale numerical features with StandardScaler
    numerical_scalers = {}
    for feat in NUMERICAL_FEATURES:
        if feat in X.columns:
            scaler = StandardScaler()
            X[feat] = scaler.fit_transform(X[[feat]])
            numerical_scalers[feat] = scaler
            print(f"[STEP 5] Scaled {feat}: mean={X[feat].mean():.4f}, std={X[feat].std():.4f}")
    
    # Train/test split (60:40 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.4, random_state=42, stratify=y_binary
    )
    
    print(f"\n[STEP 5] Train/Test Split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Testing: {X_test.shape[0]} samples")
    print(f"  Train target distribution: {np.bincount(y_train)}")
    print(f"  Test target distribution: {np.bincount(y_test)}")
    
    print(f"\n[STEP 5] VERIFICATION:")
    print(f"  Feature count: {X_train.shape[1]}")
    print(f"  Expected: 30")
    print(f"  Match: {X_train.shape[1] == 30}")
    
    return X_train, X_test, y_train, y_test, ALL_FEATURES, categorical_encoders, numerical_scalers

def save_preprocessors(encoders, scalers, save_dir):
    """Save encoders and scalers for reproducibility"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'categorical_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    with open(save_dir / 'numerical_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"[STEP 5] Saved preprocessors to {save_dir}")

def load_preprocessors(save_dir):
    """Load encoders and scalers"""
    
    save_dir = Path(save_dir)
    
    with open(save_dir / 'categorical_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open(save_dir / 'numerical_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)
    
    return encoders, scalers

if __name__ == "__main__":
    # Test the unified data preparation
    X_train, X_test, y_train, y_test, feature_names, encoders, scalers = prepare_unified_data()
    
    print(f"\n{'='*60}")
    print("[STEP 5] UNIFIED DATA PREPARATION TEST - PASSED")
    print(f"{'='*60}")
