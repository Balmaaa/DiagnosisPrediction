import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
import sys
import os

# Add paths to model directories
sys.path.append('06_Decision_Trees')
sys.path.append('07_Gradient_Boosting')
sys.path.append('08_XGBoost')

class ModelComparison:
    """Compare performance of all models for appendicitis prediction using live models"""
    
    def __init__(self):
        self.models = {}
        self.comparison_metrics = {}
        
    def load_models(self):
        """Load trained models from .pkl files"""
        
        base_path = Path(__file__).parent / "09_GUI_Application" / "saved_models"
        
        # Model files
        model_files = {
            'Decision Tree': base_path / "Decision_Tree.pkl",
            'Gradient Boosting': base_path / "Gradient_Boosting.pkl", 
            'XGBoost': base_path / "XGBoost.pkl",
            'Transformer': base_path / "Transformer.pkl"
        }
        
        for model_name, model_path in model_files.items():
            print(f"Loading {model_name}...")
            
            if model_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} from {model_path.name}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
            else:
                print(f"Model file not found: {model_path}")
    
    def load_test_data(self):
        """Load the same test data used by backend_predictor"""
        
        # Use the same data loading as backend_predictor
        sys.path.append('06_Decision_Trees')
        from decision_tree_model import DecisionTreeModel
        
        dt_wrapper = DecisionTreeModel()
        X_train, X_test, y_train, y_test = dt_wrapper.load_unified_data()
        
        return X_test, y_test
    
    def create_comparison_table(self):
        """Create comparison table using live model evaluation with 0.4 threshold"""
        
        if not self.models:
            self.load_models()
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        comparison_data = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            try:
                # Get predicted probabilities
                y_proba = model.predict_proba(X_test)
                
                # Apply 0.5 threshold (standard default)
                y_pred = (y_proba[:, 1] >= 0.5).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, pos_label=1)
                sensitivity = recall_score(y_test, y_pred, pos_label=1)
                
                # Calculate specificity
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Calculate PPV and NPV
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                comparison_data.append({
                    'Model': model_name,
                    'Dataset': 'CSV',
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'PPV': ppv,
                    'NPV': npv,
                    'True Positives': tp,
                    'True Negatives': tn,
                    'False Positives': fp,
                    'False Negatives': fn
                })
                
                print(f"  Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        print("\n" + "="*100)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*100)
        
        # Group by dataset
        for dataset in self.comparison_df['Dataset'].unique():
            print(f"\n{'='*60}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*60}")
            
            dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
            
            # Format the table
            print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Sensitivity':<12} {'Specificity':<12}")
            print("-" * 70)
            
            for _, row in dataset_df.iterrows():
                print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Sensitivity']:<12.4f} {row['Specificity']:<12.4f}")
            
            # Find best model for each metric
            best_accuracy = dataset_df.loc[dataset_df['Accuracy'].idxmax()]
            best_sensitivity = dataset_df.loc[dataset_df['Sensitivity'].idxmax()]
            best_specificity = dataset_df.loc[dataset_df['Specificity'].idxmax()]
            
            print(f"\nBest Performing Models:")
            print(f"  Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
            print(f"  Sensitivity: {best_sensitivity['Model']} ({best_sensitivity['Sensitivity']:.4f})")
            print(f"  Specificity: {best_specificity['Model']} ({best_specificity['Specificity']:.4f})")
    
    def plot_model_comparison(self, save_path=None):
        """Create visualization of model comparison"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (metric, ax) in enumerate(zip(metrics, axes.flatten())):
            # Group by dataset
            for dataset_idx, dataset in enumerate(self.comparison_df['Dataset'].unique()):
                dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
                
                x_pos = np.arange(len(dataset_df))
                width = 0.35
                
                # Create bars
                bars = ax.bar(
                    x_pos + dataset_idx * width, 
                    dataset_df[metric], 
                    width, 
                    label=dataset,
                    color=colors[dataset_idx],
                    alpha=0.8
                )
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2., 
                        height + 0.01,
                        f'{height:.3f}',
                        ha='center', 
                        va='bottom',
                        fontsize=8
                    )
            
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xticks(x_pos + width/2)
            ax.set_xticklabels(dataset_df['Model'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_feature_importance(self):
        """Analyze and compare feature importance across models"""
        
        print("\n" + "="*100)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*100)
        
        for model_name, results in self.models_data.items():
            if 'feature_importance' in results:
                print(f"\n{'='*60}")
                print(f"TOP FEATURES - {model_name.upper()}")
                print(f"{'='*60}")
                
                feature_importance = results['feature_importance']
                print(feature_importance.to_string(index=False))
                print()
    
    def generate_summary_report(self, save_path=None):
        """Generate comprehensive summary report"""
        
        if not hasattr(self, 'comparison_df'):
            self.create_comparison_table()
        
        report = []
        report.append("="*100)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("Pediatric Appendicitis Prediction System")
        report.append("="*100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        report.append("EXECUTIVE SUMMARY:")
        report.append("-" * 50)
        
        # Find best overall model
        best_overall = self.comparison_df.loc[self.comparison_df['Accuracy'].idxmax()]
        report.append(f"Best Overall Model: {best_overall['Model']} ({best_overall['Dataset']})")
        report.append(f"Best Accuracy: {best_overall['Accuracy']:.4f}")
        report.append("")
        
        # Dataset summary
        for dataset in self.comparison_df['Dataset'].unique():
            dataset_df = self.comparison_df[self.comparison_df['Dataset'] == dataset]
            
            report.append(f"DATASET: {dataset.upper()}")
            report.append("-" * 30)
            
            for _, row in dataset_df.iterrows():
                report.append(f"{row['Model']}:")
                report.append(f"  Accuracy: {row['Accuracy']:.4f}")
                report.append(f"  Sensitivity: {row['Sensitivity']:.4f}")
                report.append(f"  Specificity: {row['Specificity']:.4f}")
                report.append(f"  PPV: {row['PPV']:.4f}")
                report.append(f"  NPV: {row['NPV']:.4f}")
                report.append("")
        
        # Medical interpretation
        report.append("MEDICAL INTERPRETATION:")
        report.append("-" * 50)
        report.append("• Sensitivity (True Positive Rate): Ability to correctly identify appendicitis cases")
        report.append("• Specificity (True Negative Rate): Ability to correctly identify non-appendicitis cases")
        report.append("• PPV (Positive Predictive Value): Probability that positive prediction is correct")
        report.append("• NPV (Negative Predictive Value): Probability that negative prediction is correct")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)
        
        # Find models with best medical metrics
        best_sensitivity = self.comparison_df.loc[self.comparison_df['Sensitivity'].idxmax()]
        best_specificity = self.comparison_df.loc[self.comparison_df['Specificity'].idxmax()]
        
        report.append(f"• For screening (high sensitivity needed): {best_sensitivity['Model']} ({best_sensitivity['Sensitivity']:.4f})")
        report.append(f"• For confirmation (high specificity needed): {best_specificity['Model']} ({best_specificity['Specificity']:.4f})")
        report.append("• Consider ensemble approach for balanced performance")
        report.append("")
        
        report.append("="*100)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to: {save_path}")
        
        return report_text
    
    def save_comparison_results(self, base_filename="model_comparison"):
        """Save all comparison results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison table
        if hasattr(self, 'comparison_df'):
            csv_file = f"{base_filename}_table_{timestamp}.csv"
            self.comparison_df.to_csv(csv_file, index=False)
            print(f"Comparison table saved to: {csv_file}")
        
        # Save plots
        plot_file = f"{base_filename}_plot_{timestamp}.png"
        self.plot_model_comparison(save_path=plot_file)
        
        # Save summary report
        report_file = f"{base_filename}_report_{timestamp}.txt"
        self.generate_summary_report(save_path=report_file)
        
        return {
            'csv_file': csv_file,
            'plot_file': plot_file, 
            'report_file': report_file
        }

def main():
    """Main function for model comparison using live models"""
    
    print("="*80)
    print("MODEL COMPARISON FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("LIVE MODEL EVALUATION (threshold = 0.5)")
    print("="*80)
    
    try:
        # Initialize comparison
        comparison = ModelComparison()
        
        # Load models and evaluate live
        comparison.create_comparison_table()
        comparison.print_comparison_table()
        
        # Generate plots
        comparison.plot_model_comparison()
        
        # Generate summary report
        summary_report = comparison.generate_summary_report()
        print("\n" + summary_report)
        
        # Save all results
        saved_files = comparison.save_comparison_results()
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON COMPLETED")
        print(f"{'='*80}")
        print("Files saved:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
