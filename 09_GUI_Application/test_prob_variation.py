import sys
sys.path.append('..')

from backend_predictor import AppendicitisPredictor

def test_probability_variation():
    print("=== TESTING PROBABILITY VARIATION ===")
    
    predictor = AppendicitisPredictor()
    
    # Test different cases
    test_cases = [
        {
            'Age': 8, 'Weight': 25, 'Height': 120, 'BMI': 17.4, 'Sex': 0,
            'Neutrophil_Percentage': 50, 'Body_Temperature': 36.8,
            'Lower_Right_Abd_Pain': 0, 'Migratory_Pain': 0, 'Loss_of_Appetite': 0,
            'Nausea': 0, 'Coughing_Pain': 0, 'Dysuria': 0, 'Stool': 0,
            'Peritonitis': 0, 'Severity': 0, 'Contralateral_Rebound_Tenderness': 0,
            'Ipsilateral_Rebound_Tenderness': 0, 'Psoas_Sign': 0, 'WBC_Count': 7.5,
            'RBC_Count': 4.2, 'Hemoglobin': 12.0, 'RDW': 13.0,
            'Segmented_Neutrophils': 50, 'Thrombocyte_Count': 250, 'CRP': 2,
            'Neutrophilia': 0, 'Ketones_in_Urine': 0, 'RBC_in_Urine': 0, 'WBC_in_Urine': 0
        },
        {
            'Age': 14, 'Weight': 45, 'Height': 152, 'BMI': 19.5, 'Sex': 0,
            'Neutrophil_Percentage': 75, 'Body_Temperature': 38.5,
            'Lower_Right_Abd_Pain': 1, 'Migratory_Pain': 1, 'Loss_of_Appetite': 1,
            'Nausea': 1, 'Coughing_Pain': 0, 'Dysuria': 0, 'Stool': 0,
            'Peritonitis': 1, 'Severity': 2, 'Contralateral_Rebound_Tenderness': 1,
            'Ipsilateral_Rebound_Tenderness': 1, 'Psoas_Sign': 1, 'WBC_Count': 16.5,
            'RBC_Count': 4.8, 'Hemoglobin': 14.0, 'RDW': 15.0,
            'Segmented_Neutrophils': 80, 'Thrombocyte_Count': 350, 'CRP': 50,
            'Neutrophilia': 1, 'Ketones_in_Urine': 0, 'RBC_in_Urine': 0, 'WBC_in_Urine': 0
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        prediction, proba = predictor.predict('Transformer', test_case)
        print(f"Prediction: {prediction}")
        print(f"Probabilities: No={proba[0]:.4f}, Yes={proba[1]:.4f}")
        print(f"Confidence: {max(proba):.4f}")
    
    print("\n=== PROBABILITY VARIATION TEST COMPLETE ===")

if __name__ == "__main__":
    test_probability_variation()
