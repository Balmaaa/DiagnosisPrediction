import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class XGBoostModel:
    """XGBoost model for appendicitis prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.best_params = None
        
    def load_preprocessed_data(self):
        """Load preprocessed data from previous pipeline"""
        
        script_dir = Path(__file__).parent
        
        # Load preprocessed features and targets
        excel_x_path = script_dir.parent / "04_Preprocessing Pipeline" / "excel_preprocessed_features.csv"
        excel_y_path = script_dir.parent / "04_Preprocessing Pipeline" / "excel_target.csv"
        csv_x_path = script_dir.parent / "04_Preprocessing Pipeline" / "csv_preprocessed_features.csv"
        csv_y_path = script_dir.parent / "04_Preprocessing Pipeline" / "csv_target.csv"
        
        datasets = []
        
        # Load Excel dataset
        if excel_x_path.exists() and excel_y_path.exists():
            excel_X = pd.read_csv(excel_x_path)
            excel_y = pd.read_csv(excel_y_path, header=None)
            excel_y.columns = ['Diagnosis']
            datasets.append(('Excel', excel_X, excel_y))
            print(f"Loaded Excel dataset: {excel_X.shape}, Target: {excel_y.shape}")
        
        # Load CSV dataset
        if csv_x_path.exists() and csv_y_path.exists():
            csv_X = pd.read_csv(csv_x_path)
            csv_y = pd.read_csv(csv_y_path, header=None)
            csv_y.columns = ['Diagnosis']
            datasets.append(('CSV', csv_X, csv_y))
            print(f"Loaded CSV dataset: {csv_X.shape}, Target: {csv_y.shape}")
        
        if not datasets:
            raise FileNotFoundError("No preprocessed datasets found. Run preprocessing pipeline first.")
        
        return datasets
    
    def prepare_data(self, X, y):
        """Prepare data for XGBoost model"""
        
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
        """Perform hyperparameter tuning for XGBoost"""
        
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
        
        # Perform Grid Search
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {self.best_params}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def train_model(self, X_train, y_train, use_hyperparameter_tuning=True):
        """Train XGBoost model"""
        
        if use_hyperparameter_tuning:
            self.model = self.hyperparameter_tuning(X_train, y_train)
        else:
            # Use default parameters
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=1.0,
                colsample_bytree=1.0,
                gamma=0,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False
            )
            self.model.fit(X_train, y_train)
        
        print(f"Model trained successfully!")
        print(f"Number of estimators: {self.model.n_estimators}")
        print(f"Best score: {self.model.best_score if hasattr(self.model, 'best_score') else 'N/A'}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate XGBoost model"""
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
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
    
    def plot_training_history(self, save_path=None):
        """Plot training history for XGBoost"""
        
        if self.model is None or not hasattr(self.model, 'evals_result'):
            print("No evaluation history available!")
            return
        
        # Get evaluation results
        results = self.model.evals_result()
        
        if not results:
            print("No evaluation results found!")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot training and validation loss
        for metric in results:
            for dataset in results[metric]:
                plt.plot(results[metric][dataset], label=f'{dataset} {metric}')
        
        plt.xlabel('Iterations')
        plt.ylabel('Metric Value')
        plt.title('XGBoost Training History')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance from XGBoost"""
        
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
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """Plot feature importance"""
        
        feature_importance = self.get_feature_importance(top_n)
        
        if feature_importance is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bar chart
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - XGBoost')
        plt.gca().invert_yaxis()  # To display highest importance at top
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tree(self, tree_index=0, save_path=None):
        """Plot individual XGBoost tree"""
        
        if self.model is None:
            print("Model not trained yet!")
            return
        
        plt.figure(figsize=(20, 12))
        
        # Plot tree
        xgb.plot_tree(
            self.model, 
            tree_index=tree_index,
            feature_names=self.feature_names[:50] if len(self.feature_names) > 50 else self.feature_names,
            ax=plt.gca()
        )
        
        plt.title(f'XGBoost Tree {tree_index}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Tree plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
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
    """Main function for XGBoost model training and evaluation"""
    
    print("=" * 60)
    print("XGBOOST MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("=" * 60)
    
    try:
        # Initialize XGBoost model
        xgb_model = XGBoostModel()
        
        # Load preprocessed datasets
        datasets = xgb_model.load_preprocessed_data()
        
        for dataset_name, X, y in datasets:
            print(f"\n{'='*60}")
            print(f"TRAINING {dataset_name.upper()} DATASET")
            print(f"{'='*60}")
            
            # Prepare data
            X_array, y_array, label_encoder = xgb_model.prepare_data(X, y)
            xgb_model.label_encoder = label_encoder
            
            # Split data (60:40 as specified in paper)
            X_train, X_test, y_train, y_test = xgb_model.split_data(X_array, y_array)
            
            # Train model
            print(f"\nTraining XGBoost model...")
            xgb_model.train_model(X_train, y_train, use_hyperparameter_tuning=True)
            
            # Evaluate model
            print(f"\nEvaluating model...")
            metrics, y_pred, y_pred_proba = xgb_model.evaluate_model(X_test, y_test)
            
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS - {dataset_name.upper()} DATASET")
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
            feature_importance = xgb_model.get_feature_importance(top_n=10)
            print(f"\nTop 10 Feature Importance:")
            print(feature_importance.to_string(index=False))
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"xgboost_results_{dataset_name}_{timestamp}.pkl"
            model_file = f"xgboost_model_{dataset_name}_{timestamp}.pkl"
            feature_importance_file = f"xgboost_feature_importance_{dataset_name}_{timestamp}.png"
            
            # Save model and results
            xgb_model.save_model(model_file)
            
            results = {
                'dataset_name': dataset_name,
                'model_type': 'XGBoost',
                'best_params': xgb_model.best_params,
                'final_metrics': metrics,
                'feature_importance': feature_importance,
                'timestamp': timestamp
            }
            
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            # Generate plots
            xgb_model.plot_feature_importance(top_n=15, save_path=feature_importance_file)
            # xgb_model.plot_tree(tree_index=0, save_path=tree_plot_file)  # Disabled due to graphviz dependency
            print(f"Tree plot skipped due to graphviz dependency")
            
            print(f"\nResults saved to: {results_file}")
            print(f"Model saved to: {model_file}")
            print(f"Feature importance plot saved to: {feature_importance_file}")
        
        print(f"\n{'='*60}")
        print("XGBOOST MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
