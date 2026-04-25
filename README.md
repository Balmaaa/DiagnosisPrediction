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

## Medical Interpretation
- Sensitivity (True Positive Rate): Ability to correctly identify appendicitis cases
- Specificity (True Negative Rate): Ability to correctly identify non-appendicitis cases
- PPV (Positive Predictive Value): Probability that positive prediction is correct
- NPV (Negative Predictive Value): Probability that negative prediction is correct

## Recommendations
- For screening (high sensitivity needed): Transformer (77.20%)
- For confirmation (high specificity needed): XGBoost (95.36%)
- Consider ensemble approach for balanced performance

## Key Components

### 1. Backend Predictor (`backend_predictor.py`)
- Loads and manages pre-trained models
- Handles data preprocessing
- Provides prediction interface

### 2. GUI Application (`prediction_gui.py`)
- Interactive form for patient data input
- Real-time AI predictions with 4 model options
- Medical interpretation of results
- Features: scroll wheel support, temperature scaling for Transformer

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

### Latest Models (April 25, 2026)
| Model | Accuracy | Sensitivity | Specificity | PPV | NPV | Best Use Case |
|--------|----------|-----------|-------------|------|------|---------------|
| **XGBoost** | 85.30% | 68.91% | 95.36% | 90.11% | 83.33% | High specificity |
| **Decision Tree** | 84.98% | 73.95% | 91.75% | 84.62% | 85.17% | Balanced performance |
| **Gradient Boosting** | 84.98% | 69.75% | 94.33% | 88.30% | 83.56% | High PPV |
| **Transformer** | 77.00% | 77.20% | 71.43% | 98.68% | 10.20% | High sensitivity |

### Transformer Model Evolution
- **April 4**: 96.4% specificity (CrossEntropyLoss)
- **April 25**: 71.4% specificity (BCEWithLogitsLoss with pos_weight)
- **Trade-off**: Improved sensitivity (69.8% → 77.2%) at cost of specificity

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
python prediction_gui.py
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

### Transformer Architecture (Latest)
- **Embedding Dimension**: 128
- **Layers**: 6
- **Attention Heads**: 4 (GUI) / 8 (checkpoint)
- **Loss Function**: BCEWithLogitsLoss with pos_weight
- **Temperature Scaling**: 1.0788 (probability calibration)
- **Features**: 30 clinical + 8 missing indicators = 38 total
- **Normalization**: StandardScaler on numerical features

## Evaluation Metrics
- Accuracy, Precision, Sensitivity (Recall)
- Specificity, PPV, NPV
- F1-Score, ROC-AUC

## Ethical Considerations
- AI predictions supplement, not replace, clinical judgment
- Data privacy and confidentiality maintained
- Model limitations clearly communicated

## Model File Locations

### Latest Models (April 25, 2026)
```
09_GUI_Application/saved_models/
├── Decision_Tree.pkl
├── Gradient_Boosting.pkl  
├── XGBoost.pkl
├── Transformer.pt (with temperature scaling)
└── metadata.pkl
```

### Historical Models (April 4, 2026)
```
06_Decision_Trees/decision_tree_model_fair_20260404_193942.pkl
07_Gradient_Boosting/gradient_boosting_model_fair_20260404_194743.pkl
08_XGBoost/xgboost_model_fair_20260404_195216.pkl
05_Transformer_Model/best_advanced_transformer_model.pth
```

## Recent Changes (April 2026)

### Transformer Model Updates
1. **Loss Function**: CrossEntropyLoss → BCEWithLogitsLoss with pos_weight
2. **Architecture**: Increased embed_dim (64→128), layers (4→6)
3. **Features**: Added temperature scaling for probability calibration
4. **Performance**: Improved sensitivity (69.8%→77.2%) at cost of specificity (96.4%→71.4%)

### GUI Improvements
1. **Consolidated Models**: All models now in `saved_models/` directory
2. **Enhanced UX**: Scroll wheel support, improved layout
3. **Updated Files**: Replaced old GUI files with optimized `prediction_gui.py`

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
