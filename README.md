# Diagnosis Prediction Model for Pediatric Appendicitis using Artificial Intelligence Transformer Model

## Overview
This project develops an AI-based diagnosis prediction model for pediatric appendicitis using Transformer Models and compares it with traditional machine learning approaches (Decision Trees, Gradient Boosting, XGBoost).

## Project Structure
```
├── 01_Data Audit/              # Data quality assessment
├── 02_Target Leak Control/     # Target leakage prevention
├── 03_Feature Categorization/  # Feature classification system
├── 04_Preprocessing Pipeline/  # Data preprocessing pipeline
├── 05_Transformer_Model/       # Transformer model implementation
├── 06_Decision_Trees/          # Decision Tree models
├── 07_Gradient_Boosting/       # Gradient Boosting models
├── 08_XGBoost/                 # XGBoost models
├── 09_GUI_Application/         # User interface for predictions
├── Dataset 1/                  # First dataset (images and Excel)
├── Dataset 2/                  # Second dataset (CSV)
└── model_comparison.py         # Model comparison script
```

## Features
- **Multiple AI Models**: Transformer, Decision Tree, Gradient Boosting, XGBoost
- **Real Dataset Integration**: Uses actual pediatric appendicitis datasets
- **Interactive GUI**: User-friendly interface for medical professionals
- **Comprehensive Evaluation**: Model comparison with medical statistics metrics
- **Data Preprocessing**: Automated pipeline for data cleaning and preparation

## Key Components

### 1. Backend Predictor (`backend_predictor.py`)
- Loads and manages pre-trained models
- Handles data preprocessing
- Provides prediction interface

### 2. GUI Application (`simple_gui.py`)
- Interactive form for patient data input
- Real-time AI predictions
- Medical interpretation of results

### 3. Model Training Scripts
- Individual training scripts for each model type
- Comprehensive evaluation metrics
- Feature importance analysis

## Datasets
- **Dataset 1**: Pediatric appendicitis cases with imaging data
- **Dataset 2**: Regensburg Pediatric Appendicitis dataset (782 samples)

## Target Population
- Pediatric patients (<18 years old)
- Suspected acute appendicitis cases

## Model Performance
- **Accuracy**: Up to 86%
- **Precision**: Up to 91%
- **Recall**: Up to 75%
- **F1-Score**: Up to 82%

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages listed in requirements.txt

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
```bash
cd 09_GUI_Application
python simple_gui.py
```

## Usage
1. Launch the GUI application
2. Select desired AI model
3. Input patient data
4. Get AI-powered diagnosis prediction
5. Review medical interpretation

## User Roles
- **AI Expert**: Model training and evaluation
- **Medical Expert**: System validation and diagnosis review
- **Medical Student**: Learning and case familiarization

## Technical Specifications

### Input Features
- **Demographic**: Age, Sex, Height, Weight, BMI
- **Clinical**: Pain symptoms, nausea, appetite loss
- **Laboratory**: WBC, CRP, temperature, neutrophils
- **Imaging**: Ultrasound findings, appendix measurements
- **Scoring**: Alvarado Score, Pediatric Appendicitis Score

### Output
- **Diagnosis**: Appendicitis/No Appendicitis
- **Confidence**: Prediction probability (0-1)
- **Risk Assessment**: Clinical interpretation

## Evaluation Metrics
- Accuracy, Precision, Sensitivity (Recall)
- Specificity, PPV, NPV
- F1-Score, ROC-AUC

## Ethical Considerations
- AI predictions supplement, not replace, clinical judgment
- Data privacy and confidentiality maintained
- Model limitations clearly communicated

## Future Improvements
- Enhanced Transformer architecture
- Additional datasets for training
- Real-time clinical integration
- Mobile application development

## Citation
If you use this project in research, please cite appropriately.

## License
[Specify your license here]

## Contact
[Your contact information]
