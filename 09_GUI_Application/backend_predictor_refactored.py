"""
Refactored Backend Predictor - Clean Architecture
Imports models from their respective files instead of duplicate logic
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for model imports
sys.path.append(str(Path(__file__).parent.parent))

class AppendicitisPredictor:
    """Backend predictor that imports models from their respective files"""
    
    def __init__(self):
        self.models = {}
        self.model_classes = {}
        self.is_trained = False
        self.load_components()
    
    def load_components(self):
        """Load models from their respective model files"""
        try:
            # Import models from their dedicated files
            self._import_models()
            print("✓ Models imported successfully from model files")
        except Exception as e:
            print(f"✗ Error importing models: {e}")
            self._create_fallback_models()
    
    def _import_models(self):
        """Import model classes from their respective files"""
        
        # Import Decision Tree
        try:
            sys.path.append(str(Path(__file__).parent.parent / "06_Decision_Trees"))
            from decision_tree_model import DecisionTreeModel
            self.model_classes['Decision Tree'] = DecisionTreeModel
            print("✓ Decision Tree imported")
        except ImportError as e:
            print(f"⚠ Decision Tree import failed: {e}")
        
        # Import Gradient Boosting
        try:
            sys.path.append(str(Path(__file__).parent.parent / "07_Gradient_Boosting"))
            from gradient_boosting_model import GradientBoostingModel
            self.model_classes['Gradient Boosting'] = GradientBoostingModel
            print("✓ Gradient Boosting imported")
        except ImportError as e:
            print(f"⚠ Gradient Boosting import failed: {e}")
        
        # Import XGBoost
        try:
            sys.path.append(str(Path(__file__).parent.parent / "08_XGBoost"))
            from xgboost_model import XGBoostModel
            self.model_classes['XGBoost'] = XGBoostModel
            print("✓ XGBoost imported")
        except ImportError as e:
            print(f"⚠ XGBoost import failed: {e}")
        
        # Import Transformer
        try:
            sys.path.append(str(Path(__file__).parent.parent / "05_Transformer_Model"))
            from transformer_model import AdvancedTabularTransformer
            self.model_classes['Transformer'] = AdvancedTabularTransformer
            print("✓ Transformer imported")
        except ImportError as e:
            print(f"⚠ Transformer import failed: {e}")
    
    def _create_fallback_models(self):
        """Create fallback models if imports fail"""
        print("⚠ Using fallback model creation")
        # This would only be used if model files are missing
        pass
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models using their respective model files.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        results = {}
        
        for model_name, model_class in self.model_classes.items():
            try:
                print(f"\n{'='*60}")
                print(f"Training {model_name}...")
                print(f"{'='*60}")
                
                # Instantiate model
                model = model_class()
                
                # Train using the model's own training method
                # Models now include: class_weight, scale_pos_weight, f1 scoring
                if hasattr(model, 'train_model'):
                    model.train_model(X_train, y_train, use_hyperparameter_tuning=True)
                
                # Evaluate with threshold adjustment capability
                if hasattr(model, 'evaluate_model'):
                    # Test with default threshold (0.5)
                    metrics_default = model.evaluate_model(X_test, y_test, threshold=0.5)
                    
                    # Test with lower threshold (0.4) for better sensitivity
                    metrics_low = model.evaluate_model(X_test, y_test, threshold=0.4)
                    
                    results[model_name] = {
                        'model': model,
                        'metrics_default': metrics_default,
                        'metrics_low_threshold': metrics_low
                    }
                    
                    # Print comparison
                    self._print_threshold_comparison(model_name, metrics_default, metrics_low)
                
                self.models[model_name] = model
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        self.is_trained = len(self.models) > 0
        return results
    
    def _print_threshold_comparison(self, model_name, metrics_default, metrics_low):
        """Print comparison of default vs low threshold"""
        print(f"\n{model_name} - Threshold Comparison:")
        print(f"  Default (0.5):  Sensitivity={metrics_default.get('sensitivity', 0):.4f}, "
              f"Specificity={metrics_default.get('specificity', 0):.4f}")
        print(f"  Lower (0.4):    Sensitivity={metrics_low.get('sensitivity', 0):.4f}, "
              f"Specificity={metrics_low.get('specificity', 0):.4f}")
    
    def predict(self, model_name, X, threshold=0.5):
        """
        Make prediction using specified model.
        
        Args:
            model_name: Name of model to use
            X: Features to predict
            threshold: Classification threshold (default 0.5, can be lowered for higher sensitivity)
        
        Returns:
            predictions: Binary predictions
            probabilities: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        # Get probabilities
        if hasattr(model, 'model') and hasattr(model.model, 'predict_proba'):
            proba = model.model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"Model {model_name} doesn't support probability prediction")
        
        # Apply threshold
        predictions = (proba >= threshold).astype(int)
        
        return predictions, proba
    
    def get_model_metrics(self, model_name, X_test, y_test, threshold=0.5):
        """Get metrics for a specific model with threshold adjustment"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'evaluate_model'):
            return model.evaluate_model(X_test, y_test, threshold=threshold)
        
        return None
    
    def list_models(self):
        """List available models"""
        return list(self.models.keys())


def main():
    """Test the refactored backend predictor"""
    print("="*80)
    print("TESTING REFACTORED BACKEND PREDICTOR")
    print("="*80)
    
    predictor = AppendicitisPredictor()
    print(f"\nAvailable model classes: {list(predictor.model_classes.keys())}")


if __name__ == "__main__":
    main()
