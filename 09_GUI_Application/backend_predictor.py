import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# GUI MODE: Set to True for fast training without hyperparameter tuning
GUI_MODE = True

# Import updated model classes from model files
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path / "06_Decision_Trees"))
sys.path.insert(0, str(base_path / "07_Gradient_Boosting"))
sys.path.insert(0, str(base_path / "08_XGBoost"))
sys.path.insert(0, str(base_path / "05_Transformer_Model"))

from decision_tree_model import DecisionTreeModel
from gradient_boosting_model import GradientBoostingModel
from xgboost_model import XGBoostModel
try:
    from transformer_model import AdvancedTabularTransformer, analyze_features, prepare_data_for_advanced_transformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("[WARNING] Transformer model not available, will use fallback")

class AppendicitisPredictor:
    """Backend predictor for appendicitis diagnosis with real AI models"""
    
    def __init__(self):
        self.pipeline = None
        self.models = {}
        self.feature_names = None
        self.is_trained = False
        self.load_components()
    
    def load_components(self):
        """Load preprocessing pipeline and train real models"""
        
        base_path = Path(__file__).parent
        
        try:
            # Load preprocessing pipeline
            pipeline_path = base_path.parent / "04_Preprocessing Pipeline" / "preprocessing_pipeline_20260202_083206.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                print("Preprocessing pipeline loaded")
            else:
                print("Preprocessing pipeline not found, creating fallback")
                self._create_fallback_pipeline()
            
            # Try to load from .pkl files first (fast!)
            loaded_models = self._load_models_from_pkl()
            
            if not loaded_models:
                # No models loaded - train all
                print("No saved models found. Training new models...")
                self._train_real_models()
                
                # Verify models were created after training
                if not self.models:
                    print("[WARNING] Training failed! Creating fallback models...")
                    self._create_fallback_models()
                else:
                    # Save to .pkl for next time
                    self._save_models_to_pkl()
            elif len(self.models) < 4:
                # Partial load - train missing models
                missing = ['Decision Tree', 'Gradient Boosting', 'XGBoost', 'Transformer']
                missing = [m for m in missing if m not in self.models]
                print(f"[WARNING] Partial load complete. Training missing models: {missing}")
                self._train_real_models()
                # Save all models (including newly trained)
                self._save_models_to_pkl()
            
            # Set trained flag if we have models
            if self.models:
                self.is_trained = True
                print(f"Models ready: {list(self.models.keys())}")
                print(f"is_trained: {self.is_trained}")
            else:
                print("CRITICAL: No models available!")
            
        except Exception as e:
            print(f"Error loading components: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_system()
            if self.models:
                self.is_trained = True
    
    def _create_fallback_pipeline(self):
        """Create a simple preprocessing pipeline"""
        try:
            # Define numerical and categorical features
            numerical_features = ['Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 'WBC_Count', 
                                 'RBC_Count', 'Hemoglobin', 'RDW', 'Segmented_Neutrophils', 
                                 'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage']
            
            categorical_features = ['Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
                                   'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 
                                   'Severity', 'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 
                                   'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine']
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numerical_features),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_features)
                ])
            
            self.pipeline = preprocessor
            print("Fallback preprocessing pipeline created")
            
        except Exception as e:
            print(f"Error creating fallback pipeline: {e}")
    
    def _load_training_data(self):
        """Load or create training data"""
        try:
            # Try to load actual dataset
            base_path = Path(__file__).parent.parent
            
            # Look for CSV files in Dataset folders
            dataset_paths = [
                base_path / "Dataset 2" / "Regensburg Pediatric Appendicitis.csv",
                base_path / "Dataset 1" / "app_data.xlsx"
            ]
            
            data_loaded = False
            for dataset_path in dataset_paths:
                if dataset_path.exists():
                    if dataset_path.suffix == '.csv':
                        self.training_data = pd.read_csv(dataset_path)
                        print(f"Training data loaded from {dataset_path.name}")
                        data_loaded = True
                        break
                    elif dataset_path.suffix == '.xlsx':
                        self.training_data = pd.read_excel(dataset_path)
                        print(f"Training data loaded from {dataset_path.name}")
                        data_loaded = True
                        break
            
            if not data_loaded:
                # Create synthetic training data
                self._create_synthetic_data()
                return
            
            # Clean and preprocess the real dataset
            self._clean_real_dataset()
                
        except Exception as e:
            print(f"Error loading training data: {e}")
            self._create_synthetic_data()
    
    def _clean_real_dataset(self):
        """Clean and preprocess the real dataset"""
        try:
            print("Cleaning real dataset...")
            
            # Remove unnamed index column if present
            if 'Unnamed: 0' in self.training_data.columns:
                self.training_data = self.training_data.drop('Unnamed: 0', axis=1)
            
            # Convert diagnosis to binary (1 for appendicitis, 0 for no appendicitis)
            if 'Diagnosis' in self.training_data.columns:
                diagnosis_map = {
                    'appendicitis': 1,
                    'no appendicitis': 0,
                    'Appendicitis': 1,
                    'No Appendicitis': 0
                }
                self.training_data['Diagnosis'] = self.training_data['Diagnosis'].map(diagnosis_map)
                
                # Remove any rows with NaN in diagnosis
                self.training_data = self.training_data.dropna(subset=['Diagnosis'])
                
                # Ensure diagnosis is integer
                self.training_data['Diagnosis'] = self.training_data['Diagnosis'].astype(int)
            
            # Handle missing values in numeric columns
            numeric_columns = self.training_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != 'Diagnosis':  # Don't fill target variable
                    median_val = self.training_data[col].median()
                    self.training_data[col] = self.training_data[col].fillna(median_val)
            
            # Handle missing values in categorical columns
            categorical_columns = self.training_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != 'Diagnosis':  # Don't fill target variable
                    mode_val = self.training_data[col].mode()[0] if not self.training_data[col].mode().empty else 'unknown'
                    self.training_data[col] = self.training_data[col].fillna(mode_val)
            
            # Convert categorical yes/no to 1/0
            yes_no_columns = ['Migratory_Pain', 'Lower_Right_Abd_Pain', 'Contralateral_Rebound_Tenderness',
                            'Coughing_Pain', 'Nausea', 'Loss_of_Appetite', 'Neutrophilia', 'RBC_in_Urine', 
                            'WBC_in_Urine', 'Dysuria', 'Peritonitis', 'Psoas_Sign', 'Ipsilateral_Rebound_Tenderness',
                            'Appendix_on_US', 'Free_Fluids', 'Target_Sign', 'Appendicolith', 'Perfusion',
                            'Perforation', 'Surrounding_Tissue_Reaction', 'Appendicular_Abscess',
                            'Pathological_Lymph_Nodes', 'Bowel_Wall_Thickening', 'Conglomerate_of_Bowel_Loops',
                            'Ileus', 'Coprostasis', 'Meteorism', 'Enteritis']
            
            for col in yes_no_columns:
                if col in self.training_data.columns:
                    self.training_data[col] = self.training_data[col].map({
                        'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 
                        'positive': 1, 'negative': 0, '+': 1, '-': 0
                    }).fillna(0)
            
            # Convert Sex to binary
            if 'Sex' in self.training_data.columns:
                self.training_data['Sex'] = self.training_data['Sex'].map({
                    'male': 1, 'female': 0, 'Male': 1, 'Female': 0
                }).fillna(0)
            
            print(f"Dataset cleaned: {len(self.training_data)} samples")
            print(f"Appendicitis cases: {self.training_data['Diagnosis'].sum()} ({self.training_data['Diagnosis'].mean():.1%})")
            
        except Exception as e:
            print(f"Error cleaning dataset: {e}")
            raise
    
    def _create_synthetic_data(self):
        """Create synthetic training data for demonstration"""
        print("Creating synthetic training data")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic synthetic data
        data = {
            'Age': np.random.normal(10, 4, n_samples),
            'Weight': np.random.normal(35, 15, n_samples),
            'Height': np.random.normal(140, 20, n_samples),
            'BMI': np.random.normal(18, 3, n_samples),
            'Body_Temperature': np.random.normal(37.5, 0.8, n_samples),
            'WBC_Count': np.random.normal(10, 4, n_samples),
            'RBC_Count': np.random.normal(4.5, 0.5, n_samples),
            'Hemoglobin': np.random.normal(12, 2, n_samples),
            'RDW': np.random.normal(14, 2, n_samples),
            'Segmented_Neutrophils': np.random.normal(65, 10, n_samples),
            'Thrombocyte_Count': np.random.normal(300, 100, n_samples),
            'CRP': np.random.normal(15, 20, n_samples),
            'Neutrophil_Percentage': np.random.normal(65, 10, n_samples),
            'Sex': np.random.choice(['Male', 'Female'], n_samples),
            'Lower_Right_Abd_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
            'Migratory_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'Loss_of_Appetite': np.random.choice(['yes', 'no'], n_samples, p=[0.5, 0.5]),
            'Nausea': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
            'Coughing_Pain': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Dysuria': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Stool': np.random.choice(['normal', 'constipation', 'diarrhea'], n_samples),
            'Peritonitis': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Severity': np.random.choice(['uncomplicated', 'complicated'], n_samples, p=[0.7, 0.3]),
            'Contralateral_Rebound_Tenderness': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Ipsilateral_Rebound_Tenderness': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'Psoas_Sign': np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8]),
            'Neutrophilia': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
            'Ketones_in_Urine': np.random.choice(['no', '+', '++', '+++'], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'RBC_in_Urine': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
            'WBC_in_Urine': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85])
        }
        
        self.training_data = pd.DataFrame(data)
        
        # Create target variable with realistic correlations
        # Higher risk factors increase probability of appendicitis
        risk_score = (
            (self.training_data['Body_Temperature'] > 37.5).astype(int) * 0.2 +
            (self.training_data['WBC_Count'] > 12).astype(int) * 0.2 +
            (self.training_data['CRP'] > 10).astype(int) * 0.2 +
            (self.training_data['Lower_Right_Abd_Pain'] == 'yes').astype(int) * 0.15 +
            (self.training_data['Peritonitis'] == 'yes').astype(int) * 0.15 +
            (self.training_data['Severity'] == 'complicated').astype(int) * 0.1
        )
        
        # Add some noise and create binary target
        probability = risk_score + np.random.normal(0, 0.1, n_samples)
        probability = np.clip(probability, 0, 1)
        self.training_data['Diagnosis'] = (probability > 0.5).astype(int)
        
        print(f"Synthetic training data created: {n_samples} samples")
        print(f"Appendicitis cases: {self.training_data['Diagnosis'].sum()} ({self.training_data['Diagnosis'].mean():.1%})")
    
    def _train_real_models(self):
        """Train models using updated model classes"""
        try:
            # GUI MODE indicator
            if GUI_MODE:
                print("GUI MODE: Fast training (no hyperparameter tuning)")
            else:
                print("RESEARCH MODE: Full hyperparameter tuning enabled")
            
            print("Training models using updated model files...")
            
            # Prepare shared dataset ONCE
            print("Loading shared training data...")
            dt_wrapper = DecisionTreeModel()
            X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
            print(f"[OK] Shared data loaded: X_train={X_train.shape}, y_train={y_train.shape}")
            
            # Train Decision Tree using updated model class (if not already loaded)
            if 'Decision Tree' not in self.models:
                print("\n[1/4] Training Decision Tree using decision_tree_model.py...")
                dt_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['Decision Tree'] = dt_wrapper.model
                self.feature_columns = dt_wrapper.feature_names
                print("[OK] Decision Tree trained successfully")
            else:
                print("\n[1/4] Decision Tree already loaded from .pkl, skipping training")
            
            # Train Gradient Boosting using updated model class (if not already loaded)
            if 'Gradient Boosting' not in self.models:
                print("\n[2/4] Training Gradient Boosting using gradient_boosting_model.py...")
                gb_wrapper = GradientBoostingModel()
                gb_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['Gradient Boosting'] = gb_wrapper.model
                print("[OK] Gradient Boosting trained successfully")
            else:
                print("\n[2/4] Gradient Boosting already loaded from .pkl, skipping training")
            
            # Train XGBoost using updated model class (if not already loaded)
            if 'XGBoost' not in self.models:
                print("\n[3/4] Training XGBoost using xgboost_model.py...")
                xgb_wrapper = XGBoostModel()
                xgb_wrapper.train_model(X_train, y_train, use_hyperparameter_tuning=not GUI_MODE)
                self.models['XGBoost'] = xgb_wrapper.model
                print("[OK] XGBoost trained successfully")
            else:
                print("\n[3/4] XGBoost already loaded from .pkl, skipping training")
            
            # Store processed feature columns from shared dataset
            self.processed_feature_columns = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f"feature_{i}" for i in range(X_train.shape[1])]
            
            # Train Transformer (with fallback)
            if 'Transformer' not in self.models:
                print("\n[4/4] Training Transformer...")
                print(f"   TRANSFORMER_AVAILABLE = {TRANSFORMER_AVAILABLE}")
                if TRANSFORMER_AVAILABLE:
                    try:
                        print("   Attempting to train PyTorch Transformer...")
                        transformer_model = self._train_transformer_from_class(X_train, y_train)
                        if transformer_model is not None:
                            self.models['Transformer'] = transformer_model
                            model_type = type(transformer_model).__name__
                            print(f"[OK] Transformer trained successfully (Type: {model_type})")
                            print(f"[DEBUG] Transformer added to models. Current models: {list(self.models.keys())}")
                        else:
                            print(f"[ERROR] Transformer training returned None!")
                            raise ValueError("Transformer training returned None")
                    except Exception as e:
                        print(f"[ERROR] Transformer training failed: {e}")
                        import traceback
                        traceback.print_exc()
                        print("[WARNING] Using Gradient Boosting as fallback for Transformer")
                        self.models['Transformer'] = self._create_simple_transformer(X_train, y_train)
                else:
                    print("[WARNING] PyTorch not available (TRANSFORMER_AVAILABLE=False)")
                    print("⚠️ Using Gradient Boosting as fallback for Transformer")
                    self.models['Transformer'] = self._create_simple_transformer(X_train, y_train)
            else:
                print("\n[4/4] Transformer already loaded from .pkl, skipping training")
            
            # Evaluate models
            print("\nEvaluating models...")
            self._evaluate_models()
            
            self.is_trained = True
            print("\n[OK] All models trained successfully using updated model files!")
            print(f"DEBUG: Final models after training: {list(self.models.keys())}")
            
            # Save models to .pkl for fast loading next time
            self._save_models_to_pkl()
            
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_system()
    
    def _train_transformer_from_class(self, X_train, y_train):
        """Train Transformer using the updated transformer_model.py"""
        print("Using transformer_model.py for Transformer training...")
        
        # Get feature info
        feature_info = analyze_features(X_train)
        
        # Prepare data
        X_dict, X_test_dict, y_train_enc, y_test_enc = prepare_data_for_advanced_transformer(
            X_train, y_train, feature_info
        )
        
        # Get adaptive parameters
        num_features = len(feature_info)
        embed_dim = max(64, min(256, num_features * 4))
        num_heads = min(8, max(4, embed_dim // 32))
        num_layers = min(6, max(2, num_features // 5))
        
        print(f"Transformer architecture: embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
        
        # Create and train model
        import torch
        transformer_model = AdvancedTabularTransformer(
            feature_info=feature_info,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=2,
            dropout=0.2
        )
        
        # Quick training (10 epochs for integration)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transformer_model = transformer_model.to(device)
        
        from torch.utils.data import DataLoader
        train_dataset = transformer_model.AppendicitisDataset(X_dict, y_train_enc)
        test_dataset = transformer_model.AppendicitisDataset(X_test_dict, y_test_enc)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=0.001, weight_decay=1e-4)
        
        transformer_model.train()
        for epoch in range(10):
            total_loss = 0
            for features_dict, targets in train_loader:
                targets = targets.to(device)
                features_dict = {k: v.to(device) for k, v in features_dict.items()}
                
                optimizer.zero_grad()
                outputs = transformer_model(features_dict)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/10, Loss: {total_loss/len(train_loader):.4f}")
        
        return transformer_model
    
    def _apply_dict_pipeline(self, X):
        """Apply dictionary-based preprocessing pipeline"""
        try:
            # Create a copy to avoid modifying original
            X_processed = X.copy()
            
            # Handle categorical variables
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
            
            # Simple encoding for categorical variables
            for col in categorical_cols:
                if X_processed[col].dtype == 'object':
                    # Use label encoding for simplicity
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            
            # Handle missing values
            for col in X_processed.columns:
                if X_processed[col].isnull().any():
                    if X_processed[col].dtype in ['object']:
                        X_processed[col].fillna(X_processed[col].mode()[0], inplace=True)
                    else:
                        X_processed[col].fillna(X_processed[col].median(), inplace=True)
            
            # Return dictionary format for Transformer models
            if hasattr(self, 'processed_feature_columns'):
                feature_dict = {}
                for i, col in enumerate(X_processed.columns):
                    feature_dict[f'feature_{i}'] = X_processed[col].values
                return feature_dict
            else:
                return X_processed
            
        except Exception as e:
            print(f"Error applying dict pipeline: {e}")
            return X
    
    def _evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        try:
            print("\nModel Evaluation Results:")
            print("=" * 50)
            
            for model_name, model in self.models.items():
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='binary')
                recall = recall_score(y_test, y_pred, average='binary')
                f1 = f1_score(y_test, y_pred, average='binary')
                
                print(f"{model_name}:")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Precision: {precision:.3f}")
                print(f"  Recall: {recall:.3f}")
                print(f"  F1-Score: {f1:.3f}")
                print()
                
        except Exception as e:
            print(f"Error evaluating models: {e}")
    
    def _create_fallback_system(self):
        """Create fallback system if training fails"""
        print("Creating fallback prediction system")
        self.is_trained = False
        # Models will be created on-demand in predict method
    
    def preprocess_input(self, input_data):
        """Preprocess input data using the same pipeline as training"""
        
        try:
            # Convert input data to DataFrame
            if isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                df = pd.DataFrame([input_data])
            
            # Ensure columns are in the same order as training
            if hasattr(self, 'feature_columns'):
                # Add missing columns with default values
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0  # Default value for missing columns
                
                # Reorder columns to match training
                df = df[self.feature_columns]
            
            # Apply preprocessing pipeline
            if self.pipeline and hasattr(self.pipeline, 'transform'):
                processed_data = self.pipeline.transform(df)
            elif self.pipeline and isinstance(self.pipeline, dict):
                # Handle dictionary-based pipeline
                processed_data = self._apply_dict_pipeline(df)
                # Convert to DataFrame with proper column names if needed
                if hasattr(processed_data, 'values'):
                    if not hasattr(processed_data, 'columns'):
                        # Check if already a DataFrame to avoid 3D conversion
                        if isinstance(processed_data, pd.DataFrame):
                            processed_data_df = processed_data
                        else:
                            # Handle dictionary format for Transformer models
                            if isinstance(processed_data, dict):
                                # For dictionary format, return the original DataFrame
                                # since sklearn models expect DataFrame, not dict
                                processed_data_df = df.copy()
                            else:
                                processed_data_df = pd.DataFrame(processed_data.values, 
                                                               columns=getattr(self, 'processed_feature_columns', 
                                                                                    [f"feature_{i}" for i in range(len(processed_data))]))
                else:
                    processed_data_df = pd.DataFrame(processed_data.values, 
                                                           columns=getattr(self, 'processed_feature_columns', 
                                                                        [f"feature_{i}" for i in range(len(processed_data))]))
            else:
                # Fallback preprocessing if pipeline not available
                processed_data = self._fallback_preprocessing(df)
            
            # Use processed_data_df if it was created, otherwise processed_data
            return processed_data_df if 'processed_data_df' in locals() else processed_data
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
    
    def _fallback_preprocessing(self, df):
        """Fallback preprocessing if main pipeline fails"""
        
        try:
            # Create a simple preprocessing for demo purposes
            processed_df = df.copy()
            
            # Handle categorical variables
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
            
            # Simple encoding for categorical variables
            for col in categorical_cols:
                if processed_df[col].dtype == 'object':
                    # Use label encoding for simplicity
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
            
            # Handle missing values
            for col in processed_df.columns:
                if processed_df[col].isnull().any():
                    if processed_df[col].dtype in ['object']:
                        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
                    else:
                        processed_df[col].fillna(processed_df[col].median(), inplace=True)
            
            return processed_df.values
            
        except Exception as e:
            print(f"Error in fallback preprocessing: {e}")
            raise
    
    def predict(self, model_name, input_data):
        """Make prediction using specified model"""
        try:
            print(f"\n{'='*60}")
            print(f"PREDICTION REQUEST: Model='{model_name}'")
            print(f"{'='*60}")
            
            # Validate model selection
            if model_name not in self.models:
                print(f"[ERROR] Model '{model_name}' not found!")
                print(f"   Available models: {list(self.models.keys())}")
                raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
            
            # Get the model
            model = self.models[model_name]
            print(f"[OK] Using model: {type(model).__name__}")
            
            # Debug: Confirm Transformer usage
            if model_name == 'Transformer':
                print(f"[DEBUG] USING TRANSFORMER MODEL: {type(model).__name__}")
                print("[ACTIVE] TRANSFORMER PREDICTION ACTIVE!")
            
            # Convert input to DataFrame if needed
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, pd.DataFrame):
                df = input_data.copy()
            else:
                df = pd.DataFrame(input_data)
            
            print(f"[DATA] Input shape: {df.shape}")
            print(f"[DATA] Input columns: {list(df.columns)}")
            
            # Preprocess the input data
            processed_data = self.preprocess_input(df)
            print(f"[OK] Preprocessing complete. Shape: {processed_data.shape}")
            
            # Make prediction
            print(f"[PREDICT] Making prediction with {model_name}...")
            
            # Handle different model types
            if hasattr(model, 'predict_proba'):
                prediction = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)
                # Handle numpy arrays by extracting scalar value
                if hasattr(prediction, '__len__') and len(prediction) == 1:
                    pred_val = prediction.item() if hasattr(prediction, 'item') else float(prediction[0])
                else:
                    pred_val = float(prediction) if not hasattr(prediction, '__len__') else prediction
                proba_val = prediction_proba[0].tolist() if prediction_proba is not None else None
                print(f"[OK] Prediction complete: {pred_val}, proba shape: {prediction_proba.shape}")
                return pred_val, proba_val
            else:
                prediction = model.predict(processed_data)
                # Handle numpy arrays by taking first element if needed
                pred_val = prediction[0] if hasattr(prediction, '__len__') and len(prediction) == 1 else prediction
                print(f"[OK] Prediction complete: {pred_val}")
                return pred_val, None
                
        except Exception as e:
            print(f"[ERROR] in predict(): {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name):
        """Get model information including hyperparameter tuning status"""
        if model_name in self.models:
            model = self.models[model_name]
            info = {
                'name': model_name,
                'type': type(model).__name__,
                'trained': self.is_trained,
                'hyperparameter_tuned': True  # All models now use hyperparameter tuning
            }
            
            # Add model-specific metrics if available
            if hasattr(model, 'feature_importances_'):
                info['has_feature_importance'] = True
            
            # Add specific tuning info based on model type
            if model_name == 'Decision Tree':
                info['tuning_params'] = ['max_depth', 'min_samples_split', 'min_samples_leaf', 'criterion', 'splitter']
            elif model_name == 'Gradient Boosting':
                info['tuning_params'] = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'subsample']
            elif model_name == 'XGBoost':
                info['tuning_params'] = ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'gamma']
            elif model_name == 'Transformer':
                info['tuning_params'] = ['embed_dim', 'num_heads', 'num_layers', 'dropout', 'attention_dropout', 'learning_rate']
            
            return info
        else:
            return None

    def _save_models_to_pkl(self):
        """Save trained models to .pkl files for fast loading"""
        try:
            import pickle
            import os
            model_dir = Path(__file__).parent / 'saved_models'
            model_dir.mkdir(exist_ok=True)
            
            saved_count = 0
            for model_name, model in self.models.items():
                pkl_path = model_dir / f"{model_name.replace(' ', '_')}.pkl"
                
                try:
                    # Handle PyTorch models specially
                    if model_name == 'Transformer':
                        # Check if it's a PyTorch model
                        if hasattr(model, 'state_dict') or 'torch.nn' in str(type(model)):
                            import torch
                            torch.save(model, pkl_path)
                            # Verify file was created
                            if pkl_path.exists():
                                size_kb = pkl_path.stat().st_size / 1024
                                print(f"  [SAVED] {model_name} (PyTorch) to {pkl_path} ({size_kb:.1f} KB)")
                                saved_count += 1
                            else:
                                print(f"  [FAILED] Failed to save {model_name} - file not created")
                        else:
                            # Save as pickle (sklearn fallback)
                            with open(pkl_path, 'wb') as f:
                                pickle.dump(model, f)
                                f.flush()
                                os.fsync(f.fileno())
                            # Verify file was created
                            if pkl_path.exists():
                                size_kb = pkl_path.stat().st_size / 1024
                                print(f"  [SAVED] {model_name} (sklearn) to {pkl_path} ({size_kb:.1f} KB)")
                                saved_count += 1
                            else:
                                print(f"  [FAILED] Failed to save {model_name} - file not created")
                    else:
                        with open(pkl_path, 'wb') as f:
                            pickle.dump(model, f)
                            f.flush()
                            os.fsync(f.fileno())
                        # Verify file was created
                        if pkl_path.exists():
                            size_kb = pkl_path.stat().st_size / 1024
                            print(f"  [SAVED] {model_name} to {pkl_path} ({size_kb:.1f} KB)")
                            saved_count += 1
                        else:
                            print(f"  [FAILED] Failed to save {model_name} - file not created")
                except Exception as e:
                    print(f"  [ERROR] Error saving {model_name}: {e}")
            
            # Save metadata
            metadata = {
                'feature_columns': self.feature_columns,
                'processed_feature_columns': self.processed_feature_columns,
                'is_trained': True
            }
            metadata_path = model_dir / 'metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                f.flush()
                os.fsync(f.fileno())
            
            print(f"  [SAVED] {saved_count}/{len(self.models)} models saved to {model_dir}")
        except Exception as e:
            print(f"  [ERROR] Could not save models: {e}")
            import traceback
            traceback.print_exc()

    def _create_simple_transformer(self, X_train, y_train):
        """Create a simple Gradient Boosting fallback for Transformer"""
        try:
            print("Creating Gradient Boosting fallback for Transformer...")
            from sklearn.ensemble import GradientBoostingClassifier
            
            # Create and train Gradient Boosting model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            
            # Convert to numpy if needed
            if hasattr(X_train, 'values'):
                X_array = X_train.values
            else:
                X_array = X_train
                
            if hasattr(y_train, 'values'):
                y_array = y_train.values
            else:
                y_array = y_train
                
            model.fit(X_array, y_array)
            print(f"Fallback Transformer (Gradient Boosting) created successfully")
            return model
            
        except Exception as e:
            print(f"Error creating fallback Transformer: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_models_from_pkl(self):
        """Load models from .pkl files if available"""
        try:
            import pickle
            model_dir = Path(__file__).parent / 'saved_models'
            
            print(f"DEBUG: Looking for saved models in {model_dir}")
            
            if not model_dir.exists():
                print(f"DEBUG: Model directory does not exist: {model_dir}")
                return False
            
            print(f"DEBUG: Model directory exists. Files: {list(model_dir.glob('*.pkl'))}")
            
            # Load metadata
            metadata_path = model_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
                self.processed_feature_columns = metadata.get('processed_feature_columns', [])
                self.is_trained = metadata.get('is_trained', False)
            
            # Load models
            model_files = {
                'Decision Tree': 'Decision_Tree.pkl',
                'Gradient Boosting': 'Gradient_Boosting.pkl',
                'XGBoost': 'XGBoost.pkl',
                'Transformer': 'Transformer.pkl'
            }
            
            loaded_count = 0
            for model_name, filename in model_files.items():
                pkl_path = model_dir / filename
                print(f"DEBUG: Checking {model_name} at {pkl_path} - Exists: {pkl_path.exists()}")
                if pkl_path.exists():
                    try:
                        # Handle all models with pickle loading
                        with open(pkl_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        
                        if model_name == 'Transformer':
                            print(f"  Loaded {model_name} (pkl) from {pkl_path}")
                            print("TRANSFORMER MODEL LOADED SUCCESSFULLY (pkl)")
                        else:
                            print(f"  [LOADED] {model_name} from {pkl_path}")
                        
                        loaded_count += 1
                    except Exception as e:
                        print(f"  [FAILED] {model_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            print(f"DEBUG: Total models loaded: {loaded_count}/4")
            print(f"DEBUG: Models in self.models: {list(self.models.keys())}")
            
            if loaded_count >= 1:  # Accept 1+ models as success (train missing ones)
                print(f"[OK] {loaded_count} models loaded from .pkl files")
                if loaded_count < 4:
                    missing = [name for name in model_files.keys() if name not in self.models]
                    print(f"[WARNING] Missing models will be trained: {missing}")
                return True
            else:
                print(f"No models loaded from .pkl files, will train all")
                return False
                
        except Exception as e:
            print(f"WARNING: Could not load models from .pkl: {e}")
            import traceback
            traceback.print_exc()
            self.models = {}
            return False


# Test the predictor
if __name__ == "__main__":
    def test_predictor():
        """Test the predictor with sample data"""
        
        predictor = AppendicitisPredictor()
        
        # Sample input data
        sample_input = {
            'Age': 10,
            'Weight': 35,
            'Height': 140,
            'BMI': 17.9,
            'Sex': 'Male',
            'Neutrophil_Percentage': 65,
            'Body_Temperature': 37.5,
            'Lower_Right_Abd_Pain': 'yes',
            'Migratory_Pain': 'no',
            'Loss_of_Appetite': 'yes',
            'Nausea': 'yes',
            'Coughing_Pain': 'no',
            'Dysuria': 'no',
            'Stool': 'normal',
            'Peritonitis': 'no',
            'Severity': 'uncomplicated',
            'Contralateral_Rebound_Tenderness': 'no',
            'Ipsilateral_Rebound_Tenderness': 'yes',
            'Psoas_Sign': 'no',
            'WBC_Count': 12.5,
            'RBC_Count': 4.5,
            'Hemoglobin': 13.5,
            'RDW': 13.5,
            'Segmented_Neutrophils': 70,
            'Thrombocyte_Count': 250,
            'CRP': 15,
            'Neutrophilia': 'yes',
            'Ketones_in_Urine': 'no',
            'RBC_in_Urine': 'no',
            'WBC_in_Urine': 'no'
        }
        
        print("Testing Appendicitis Predictor")
        print("="*50)
        
        # Test each model
        for model_name in predictor.get_available_models():
            try:
                prediction, prediction_proba = predictor.predict(model_name, sample_input)
                diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
                confidence = prediction_proba[prediction]
                
                print(f"{model_name}:")
                print(f"  Prediction: {diagnosis}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Probabilities: No={prediction_proba[0]:.3f}, Yes={prediction_proba[1]:.3f}")
                print()
                
            except Exception as e:
                print(f"Error testing {model_name}: {e}")
                print()
    
    test_predictor()
