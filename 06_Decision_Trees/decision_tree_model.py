import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

class DecisionTreeModel:
    """Decision Tree model for appendicitis prediction"""
    
    def __init__(self):
        print("USING UPDATED DECISION TREE MODEL")
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.best_params = None
        
    def load_unified_data(self):
        """Load unified data for fair comparison"""
        
        # Import unified data preprocessing
        import sys
        import os
        # Add project root to path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from unified_data_preprocessing import prepare_unified_data
        
        # Load unified data
        X_train, X_test, y_train, y_test, feature_names, encoders, scalers = prepare_unified_data('CSV')
        
        # Store encoders and scalers
        self.categorical_encoders = encoders
        self.numerical_scalers = scalers
        self.feature_names = feature_names
        
        return X_train, X_test, y_train, y_test
    
    def prepare_data(self, X, y):
        """Prepare data for Decision Tree model"""
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values.flatten()
        
        # Ensure both arrays have the same length
        min_length = min(len(X_array), len(y_array))
        X_array = X_array[:min_length]
        y_array = y_array[:min_length]
        
        # Find indices where target is not NaN
        valid_indices = ~pd.isna(y_array)
        
        # Filter both features and targets
        X_clean = X_array[valid_indices]
        y_clean = y_array[valid_indices]
        
        # Encode target variable
        label_encoder = LabelEncoder()
        
        # Convert to binary: appendicitis vs no appendicitis
        y_binary = np.array([1 if str(x).lower() == 'appendicitis' else 0 for x in y_clean])
        
        y_encoded = label_encoder.fit_transform(y_binary)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        print(f"Data after cleaning: {X_clean.shape}")
        print(f"Target distribution: {np.bincount(y_encoded)}")
        print(f"Target classes: {label_encoder.classes_}")
        
        return X_clean, y_encoded, label_encoder
    
    def split_data(self, X, y, test_size=0.4, random_state=42):
        """Split data into training and testing sets (60:40 ratio as per paper)"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        print(f"Training target distribution: {np.bincount(y_train)}")
        print(f"Testing target distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for Decision Tree"""
        
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid - CONSERVATIVE to prevent overfitting
        param_grid = {
            'criterion': ['gini'],
            'max_depth': [3, 4, 5],  # Limited to shallow trees
            'min_samples_split': [10, 20, 30],  # Higher values
            'min_samples_leaf': [5, 10, 15],  # Higher values
            'max_features': ['sqrt'],  # Fixed to sqrt
            'class_weight': ['balanced']
        }
        
        # Create Decision Tree classifier with fixed seed
        dt = DecisionTreeClassifier(random_state=SEED)
        
        # Perform Grid Search - CHANGED: scoring='f1' to balance precision/recall
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        # TUNING EFFECTIVENESS DIAGNOSTICS
        print(f"\n{'='*60}")
        print("TUNING EFFECTIVENESS ANALYSIS")
        print(f"{'='*60}")
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Check if best params differ significantly from defaults
        default_depth = 3
        best_depth = self.best_params.get('max_depth', default_depth)
        depth_differs = best_depth != default_depth if best_depth is not None else True
        
        print(f"\nParameter Impact Analysis:")
        print(f"  - max_depth: {best_depth} (default would be ~{default_depth})")
        print(f"  - Parameters differ from default: {depth_differs}")
        
        # Show all CV results summary
        cv_results = grid_search.cv_results_
        mean_scores = cv_results['mean_test_score']
        print(f"  - Score range: {mean_scores.min():.4f} to {mean_scores.max():.4f}")
        print(f"  - Score std: {mean_scores.std():.4f}")
        
        if mean_scores.std() < 0.01:
            print(f"  WARNING: Low score variance - tuning may be ineffective")
            print(f"  WARNING: Suggests: Dataset bottleneck or parameter space too narrow")
        else:
            print(f"  OK: Score variance healthy - tuning is effective")
        print(f"{'='*60}\n")
        
        return self.model
    
    def train_model(self, X_train, y_train, use_hyperparameter_tuning=True):
        """Train Decision Tree model"""
        
        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters - FIXED: stronger regularization to prevent overfitting
            self.model = DecisionTreeClassifier(
                criterion='gini',
                max_depth=4,  # Reduced from 10 to prevent overfitting
                min_samples_split=10,  # Increased from 5
                min_samples_leaf=5,  # Increased from 2
                max_features='sqrt',  # Added to limit features per split
                class_weight='balanced',
                random_state=SEED
            )
            self.model.fit(X_train, y_train)
        
        print(f"Model trained successfully!")
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """Evaluate Decision Tree model with optional threshold adjustment"""
        
        # Get predicted probabilities
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Apply custom threshold for better sensitivity (default 0.5, can be lowered to 0.3-0.4)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        sensitivity = recall_score(y_test, y_pred, average='binary')  # Sensitivity = Recall
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        return metrics, y_pred, y_pred_proba
    
    def calculate_medical_metrics(self, y_true, y_pred):
        """Calculate medical-specific evaluation metrics"""
        
        # Convert to numpy arrays if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        sensitivity = recall_score(y_true, y_pred, average='binary')  # Sensitivity = Recall
        
        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate PPV and NPV
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def visualize_tree(self, max_depth=3, save_path=None):
        """Visualize the Decision Tree"""
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        plt.figure(figsize=(20, 12))
        
        # Limit feature names for visualization to avoid index error
        viz_feature_names = self.feature_names[:100] if len(self.feature_names) > 100 else self.feature_names
        
        # Plot tree with limited depth for visualization
        plot_tree(
            self.model, 
            max_depth=max_depth,
            feature_names=viz_feature_names,
            class_names=['No Appendicitis', 'Appendicitis'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        plt.title(f"Decision Tree (Depth {max_depth} visualization)", fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from Decision Tree"""
        
        if self.model is None:
            print("Model not trained yet!")
            return None
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df.head(top_n)
    
    def save_model(self, filepath):
        """Save the trained model"""
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params')
        
        print(f"Model loaded from: {filepath}")
        return self.model

def main():
    """Main function for Decision Tree model training and evaluation"""
    
    print("=" * 60)
    print("DECISION TREE MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("[FAIR COMPARISON - UNIFIED DATA]")
    print("=" * 60)
    
    try:
        # Initialize Decision Tree model
        dt_model = DecisionTreeModel()
        
        # Load unified data (same as all other models)
        print(f"\n{'='*60}")
        print("LOADING UNIFIED DATA")
        print(f"{'='*60}")
        X_train, X_test, y_train, y_test = dt_model.load_unified_data()
        
        print(f"\n{'='*60}")
        print(f"TRAINING DECISION TREE MODEL")
        print(f"{'='*60}")
        
        # Train model
        dt_model.train_model(X_train, y_train, use_hyperparameter_tuning=True)
        
        # DEBUG: Verify tuned parameters are actually used
        print(f"\n{'='*60}")
        print("MODEL PARAMETER VERIFICATION")
        print(f"{'='*60}")
        if dt_model.best_params:
            print(f"Best parameters from tuning: {dt_model.best_params}")
            print(f"Model is using tuned parameters: {dt_model.model.get_params()['max_depth'] != 3}")
        else:
            print("WARNING: No tuned parameters found - using defaults!")
        print(f"Final Model Parameters: max_depth={dt_model.model.get_params()['max_depth']}, "
              f"min_samples_split={dt_model.model.get_params()['min_samples_split']}, "
              f"min_samples_leaf={dt_model.model.get_params()['min_samples_leaf']}")
        print(f"{'='*60}")
        
        # Evaluate model
        print(f"\nEvaluating model...")
        metrics, y_pred, y_pred_proba = dt_model.evaluate_model(X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"PPV (Positive Predictive Value): {metrics['ppv']:.4f}")
        print(f"NPV (Negative Predictive Value): {metrics['npv']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives: {metrics['tp']}")
        print(f"  True Negatives: {metrics['tn']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        
        # Feature importance
        feature_importance = dt_model.get_feature_importance(top_n=10)
        print(f"\nTop 10 Feature Importance:")
        print(feature_importance.to_string(index=False))
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"decision_tree_results_fair_{timestamp}.pkl"
        model_file = f"decision_tree_model_fair_{timestamp}.pkl"
        
        # Save model and results
        dt_model.save_model(model_file)
        
        results = {
            'dataset_name': 'CSV',
            'model_type': 'Decision Tree',
            'data_type': 'unified_30_features',
            'best_params': dt_model.best_params,
            'final_metrics': metrics,
            'feature_importance': feature_importance,
            'timestamp': timestamp
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Model saved to: {model_file}")
        
        print(f"\n{'='*60}")
        print("DECISION TREE MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
