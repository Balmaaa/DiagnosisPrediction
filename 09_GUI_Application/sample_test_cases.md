# Sample Test Case for Pediatric Appendicitis Prediction System

## Test Case 1: High-Risk Appendicitis Case

### Patient Profile
- **Name**: Test Patient A
- **Age**: 12 years
- **Sex**: Male
- **Weight**: 45 kg
- **Height**: 152 cm
- **BMI**: 19.5

### Clinical Presentation
- **Chief Complaint**: Abdominal pain for 24 hours
- **Body Temperature**: 38.2°C (elevated)
- **Lower Right Abdominal Pain**: Yes
- **Migratory Pain**: Yes (started around umbilicus, moved to RLQ)
- **Loss of Appetite**: Yes
- **Nausea**: Yes
- **Peritonitis**: Yes (rebound tenderness present)
- **Severity**: Complicated
- **Contralateral Rebound Tenderness**: No
- **Ipsilateral Rebound Tenderness**: Yes
- **Psoas Sign**: Positive

### Laboratory Results
- **WBC Count**: 15.8 ×10^9/L (elevated)
- **RBC Count**: 4.6 ×10^12/L
- **Hemoglobin**: 13.8 g/dL
- **RDW**: 14.2%
- **Segmented Neutrophils**: 78% (elevated)
- **Thrombocyte Count**: 285 ×10^9/L
- **CRP**: 45 mg/L (significantly elevated)
- **Neutrophilia**: Yes
- **Ketones in Urine**: No
- **RBC in Urine**: No
- **WBC in Urine**: No

### Imaging Findings
- **US Performed**: Yes
- **US Number**: 2
- **Appendix on US**: Yes
- **Appendix Diameter**: 9.2 mm (enlarged >6mm)
- **Appendix Wall Layers**: Partially raised
- **Appendicolith**: Suspected
- **Appendicular Abscess**: No
- **Bowel Wall Thickening**: Yes
- **Conglomerate of Bowel Loops**: No
- **Coprostasis**: Yes
- **Enteritis**: No
- **Free Fluids**: Yes
- **Ileus**: No
- **Meteorism**: Yes
- **Perforation**: No
- **Perfusion**: Reduced
- **Target Sign**: Yes
- **Surrounding Tissue Reaction**: Yes
- **Pathological Lymph Nodes**: Yes
- **Lymph Nodes Location**: Mesenterial
- **Abscess Location**: None

### Expected Results
- **Diagnosis**: Appendicitis (High probability)
- **Risk Factors**: Multiple positive indicators
- **Recommended Action**: Immediate surgical consultation
- **Confidence**: High (>80%)

---

## Test Case 2: Low-Risk Non-Appendicitis Case

### Patient Profile
- **Name**: Test Patient B
- **Age**: 8 years
- **Sex**: Female
- **Weight**: 28 kg
- **Height**: 128 cm
- **BMI**: 17.1

### Clinical Presentation
- **Chief Complaint**: Abdominal discomfort
- **Body Temperature**: 37.1°C (normal)
- **Lower Right Abdominal Pain**: No
- **Migratory Pain**: No
- **Loss of Appetite**: No
- **Nausea**: No
- **Peritonitis**: No
- **Severity**: Uncomplicated
- **Contralateral Rebound Tenderness**: No
- **Ipsilateral Rebound Tenderness**: No
- **Psoas Sign**: Negative

### Laboratory Results
- **WBC Count**: 8.2 ×10^9/L (normal)
- **RBC Count**: 4.3 ×10^12/L
- **Hemoglobin**: 12.5 g/dL
- **RDW**: 12.8%
- **Segmented Neutrophils**: 55% (normal)
- **Thrombocyte Count**: 245 ×10^9/L
- **CRP**: 4 mg/L (normal)
- **Neutrophilia**: No
- **Ketones in Urine**: No
- **RBC in Urine**: No
- **WBC in Urine**: No

### Imaging Findings
- **US Performed**: Yes
- **US Number**: 1
- **Appendix on US**: Yes
- **Appendix Diameter**: 4.8 mm (normal <6mm)
- **Appendix Wall Layers**: Intact
- **Appendicolith**: No
- **Appendicular Abscess**: No
- **Bowel Wall Thickening**: No
- **Conglomerate of Bowel Loops**: No
- **Coprostasis**: No
- **Enteritis**: No
- **Free Fluids**: No
- **Ileus**: No
- **Meteorism**: No
- **Perforation**: No
- **Perfusion**: Normal
- **Target Sign**: No
- **Surrounding Tissue Reaction**: No
- **Pathological Lymph Nodes**: No
- **Lymph Nodes Location**: None
- **Abscess Location**: None

### Expected Results
- **Diagnosis**: No Appendicitis (Low probability)
- **Risk Factors**: Minimal to none
- **Recommended Action**: Continue monitoring, consider alternative diagnoses
- **Confidence**: High (>80%)

---

## Test Case 3: Borderline Case

### Patient Profile
- **Name**: Test Patient C
- **Age**: 14 years
- **Sex**: Male
- **Weight**: 52 kg
- **Height**: 165 cm
- **BMI**: 19.1

### Clinical Presentation
- **Chief Complaint**: Mild abdominal pain
- **Body Temperature**: 37.6°C (slightly elevated)
- **Lower Right Abdominal Pain**: Yes
- **Migratory Pain**: No
- **Loss of Appetite**: Yes
- **Nausea**: Yes
- **Peritonitis**: No
- **Severity**: Uncomplicated
- **Contralateral Rebound Tenderness**: No
- **Ipsilateral Rebound Tenderness**: No
- **Psoas Sign**: Negative

### Laboratory Results
- **WBC Count**: 11.5 ×10^9/L (mildly elevated)
- **RBC Count**: 4.8 ×10^12/L
- **Hemoglobin**: 14.2 g/dL
- **RDW**: 13.5%
- **Segmented Neutrophils**: 65% (mildly elevated)
- **Thrombocyte Count**: 268 ×10^9/L
- **CRP**: 12 mg/L (mildly elevated)
- **Neutrophilia**: No
- **Ketones in Urine**: No
- **RBC in Urine**: No
- **WBC in Urine**: No

### Imaging Findings
- **US Performed**: Yes
- **US Number**: 1
- **Appendix on US**: Yes
- **Appendix Diameter**: 6.5 mm (borderline)
- **Appendix Wall Layers**: Intact
- **Appendicolith**: No
- **Appendicular Abscess**: No
- **Bowel Wall Thickening**: No
- **Conglomerate of Bowel Loops**: No
- **Coprostasis**: No
- **Enteritis**: No
- **Free Fluids**: No
- **Ileus**: No
- **Meteorism**: No
- **Perforation**: No
- **Perfusion**: Normal
- **Target Sign**: No
- **Surrounding Tissue Reaction**: No
- **Pathological Lymph Nodes**: No
- **Lymph Nodes Location**: None
- **Abscess Location**: None

### Expected Results
- **Diagnosis**: Borderline (50-70% probability)
- **Risk Factors**: Some positive indicators
- **Recommended Action**: Close observation, repeat examination
- **Confidence**: Moderate (60-70%)

---

## Testing Instructions

### How to Use These Test Cases:

1. **Launch the GUI Application**
   ```bash
   cd "09_GUI_Application"
   python simple_gui.py
   ```

2. **Select Test Case**
   - Choose one of the three test cases above
   - Select a model (recommend starting with XGBoost)

3. **Input Data**
   - Enter the values from the test case into the GUI form
   - Ensure all fields are filled correctly

4. **Run Prediction**
   - Click "Predict Diagnosis"
   - Review the results and compare with expected outcomes

5. **Model Comparison**
   - Test the same case with different models
   - Compare predictions and confidence scores
   - Note any variations between models

### Expected Model Behaviors:

**Test Case 1 (High Risk):**
- All models should predict "Appendicitis"
- Confidence scores should be high (>80%)
- XGBoost and Gradient Boosting expected to perform best

**Test Case 2 (Low Risk):**
- All models should predict "No Appendicitis"
- Confidence scores should be high (>80%)
- Transformer model expected to have high specificity

**Test Case 3 (Borderline):**
- Predictions may vary between models
- Confidence scores should be moderate (60-70%)
- Good test case for model comparison

### Validation Criteria:

✅ **Successful Test**: Prediction matches expected diagnosis with reasonable confidence
✅ **Model Consistency**: Similar predictions across different models for clear cases
✅ **Medical Plausibility**: Results align with clinical expectations
✅ **Error Handling**: Graceful handling of edge cases and invalid inputs

---

## Additional Test Scenarios

### Edge Cases to Test:

1. **Empty Fields**: Test with missing required data
2. **Invalid Values**: Test with negative ages, impossible temperatures
3. **Age Limits**: Test with patients >18 years (should show warning)
4. **Extreme Values**: Very high WBC, very low BMI
5. **Model Switching**: Test changing models with same data

### Performance Tests:

1. **Response Time**: Measure prediction processing time
2. **Memory Usage**: Monitor application memory consumption
3. **Concurrent Use**: Test multiple rapid predictions
4. **Long Running**: Test application stability over extended use

---

**These test cases provide comprehensive validation of the GUI application functionality and model performance across different clinical scenarios.**
