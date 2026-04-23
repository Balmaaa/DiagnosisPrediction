"""
Threshold Optimization for Clinical Safety
Find optimal threshold maintaining sensitivity ≥ 0.90 while maximizing specificity
"""

import sys
import os
sys.path.append('..')
sys.path.append('../06_Decision_Trees')

import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from decision_tree_model import DecisionTreeModel

def optimize_threshold():
    """Find optimal threshold for clinical safety"""
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR CLINICAL SAFETY")
    print("Constraint: Sensitivity ≥ 0.90, Maximize Specificity")
    print("=" * 80)
    
    # Load models and test data
    base_path = Path(__file__).parent / "saved_models"
    
    # Load XGBoost as representative model
    model_path = base_path / "XGBoost.pkl"
    model = joblib.load(model_path)
    
    # Load test data
    dt_wrapper = DecisionTreeModel()
    X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
    
    # Define threshold range
    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    
    print(f"\nStep 1: Threshold Results Table")
    print("-" * 50)
    print(f"{'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12}")
    print("-" * 50)
    
    results = []
    valid_thresholds = []
    
    for threshold in thresholds:
        # Get predictions
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, pos_label=1)
        sensitivity = recall_score(y_test, y_pred, pos_label=1)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision
        })
        
        print(f"{threshold:<10.2f} {sensitivity:<12.4f} {specificity:<12.4f} {precision:<12.4f}")
        
        # Apply sensitivity constraint
        if sensitivity >= 0.90:
            valid_thresholds.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision
            })
    
    print(f"\nStep 2: Valid Thresholds (Sensitivity ≥ 0.90)")
    print("-" * 50)
    
    if not valid_thresholds:
        print("No thresholds meet sensitivity ≥ 0.90 constraint!")
        return None
    
    for vt in valid_thresholds:
        print(f"Threshold {vt['threshold']:.2f}: Sensitivity={vt['sensitivity']:.4f}, Specificity={vt['specificity']:.4f}")
    
    # Select best threshold (highest specificity)
    best_threshold_data = max(valid_thresholds, key=lambda x: x['specificity'])
    best_threshold = best_threshold_data['threshold']
    
    print(f"\nStep 3: Selected Threshold")
    print("-" * 50)
    print(f"Selected Threshold: {best_threshold}")
    print(f"Sensitivity: {best_threshold_data['sensitivity']:.4f}")
    print(f"Specificity: {best_threshold_data['specificity']:.4f}")
    print(f"Precision: {best_threshold_data['precision']:.4f}")
    
    return best_threshold

if __name__ == "__main__":
    from pathlib import Path
    best_threshold = optimize_threshold()
    
    if best_threshold:
        print(f"\nStep 4: Clinical Justification")
        print("-" * 50)
        print(f"Threshold {best_threshold} provides:")
        print(f"  • Sensitivity ≥ 0.90: Ensures patient safety (missed cases minimized)")
        print(f"  • Highest Specificity: Reduces false positives and unnecessary treatments")
        print(f"  • Balanced Performance: Optimizes clinical decision-making")
        
        print(f"\nStep 5: Apply Globally")
        print("-" * 50)
        print(f"Update backend_predictor.py and model_comparison.py with threshold = {best_threshold}")
    else:
        print("\nERROR: No suitable threshold found!")
