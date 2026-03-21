import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_output_files():
    """Setup output files for saving preprocessing results"""
    import os
    from pathlib import Path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"preprocessing_pipeline_{timestamp}.txt"
    pkl_filename = f"preprocessing_pipeline_{timestamp}.pkl"
    
    script_dir = Path(__file__).parent
    txt_path = script_dir / txt_filename
    pkl_path = script_dir / pkl_filename
    
    txt_file = open(txt_path, 'w', encoding='utf-8')
    
    return txt_file, str(txt_path), str(pkl_path)

def print_and_save(text, output_file):
    print(text)
    output_file.write(text + '\n')

def load_datasets_and_classification():
    import os
    from pathlib import Path
    import glob
    
    current_dir = Path.cwd()
    if current_dir.name == "04_Preprocessing Pipeline":
        excel_path = "../Dataset 1/app_data.xlsx"
        csv_path = "../Dataset 2/Regensburg Pediatric Appendicitis.csv"
        classification_pattern = "../03_Feature Categorization/feature_classification_*.csv"
    else:
        excel_path = "Dataset 1/app_data.xlsx"
        csv_path = "Dataset 2/Regensburg Pediatric Appendicitis.csv"
        classification_pattern = "03_Feature Categorization/feature_classification_*.csv"
    
    df_excel = pd.read_excel(excel_path)
    df_csv = pd.read_csv(csv_path)
    
    print_and_save(f"[OK] Loaded Excel dataset: {excel_path}", output_file)
    print_and_save(f"[OK] Loaded CSV dataset: {csv_path}", output_file)
    
    classification_files = glob.glob(classification_pattern)
    if not classification_files:
        raise FileNotFoundError("No classification CSV file found. Run feature_categorization.py first.")
    
    latest_classification = max(classification_files)
    df_classification = pd.read_csv(latest_classification)
    print_and_save(f"[OK] Loaded classification from: {latest_classification}", output_file)
    
    return df_excel, df_csv, df_classification

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, excluded_categories):
        self.excluded_categories = excluded_categories
        self.columns_to_drop_ = None
    
    def fit(self, X, y=None):
        self.columns_to_drop_ = []
        for col in X.columns:
            if col in self.excluded_categories:
                self.columns_to_drop_.append(col)
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_drop_:
            X_transformed = X_transformed.drop(columns=self.columns_to_drop_)
        return X_transformed

class TargetSeparator(BaseEstimator, TransformerMixin):   
    def __init__(self, target_column):
        self.target_column = target_column
        self.y_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.target_column in X.columns:
            self.y_ = X[self.target_column].copy()
            X_transformed = X.drop(columns=[self.target_column])
        else:
            self.y_ = None
            X_transformed = X.copy()
        return X_transformed
    
    def get_target(self):
        return self.y_

def create_preprocessing_pipeline(df_classification):
    print_and_save("\nCreating preprocessing pipeline...", output_file)
    
    excluded_categories = ['Administrative (Excluded)', 'Outcome-Derived (Excluded)', 'Scoring System (Excluded)']
    excluded_df = df_classification[df_classification['Category'].isin(excluded_categories)]
    excluded_columns = excluded_df['Column_Name'].tolist()
    
    print_and_save(f"Excluded columns ({len(excluded_columns)}):", output_file)
    for col in excluded_columns:
        print_and_save(f"  - {col}", output_file)
    
    outcome_df = df_classification[df_classification['Category'] == 'Outcome']
    target_column = outcome_df['Column_Name'].iloc[0] if len(outcome_df) > 0 else None
    
    if not target_column:
        raise ValueError("No target column found in classification")
    
    print_and_save(f"Target column: {target_column}", output_file)
    
    pipeline = Pipeline([
        ('drop_columns', ColumnDropper(excluded_columns)),
        ('separate_target', TargetSeparator(target_column)),
        ('preprocess', ColumnTransformer(
            transformers=[
                ('numeric', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), []),
                
                ('categorical', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), [])
            ],
            remainder='drop'
        ))
    ])
    
    return pipeline, target_column

def identify_column_types(df, df_classification, target_column):
    print_and_save("\nIdentifying column types...", output_file)
    
    excluded_categories = ['Administrative (Excluded)', 'Outcome-Derived (Excluded)', 'Scoring System (Excluded)', 'Outcome']
    feature_df = df_classification[~df_classification['Category'].isin(excluded_categories)]
    feature_columns = feature_df['Column_Name'].tolist()
    
    available_columns = [col for col in feature_columns if col in df.columns]
    
    numeric_columns = []
    categorical_columns = []
    
    for col in available_columns:
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)
    
    print_and_save(f"Numeric features ({len(numeric_columns)}):", output_file)
    for col in numeric_columns:
        print_and_save(f"  - {col} ({df[col].dtype})", output_file)
    
    print_and_save(f"Categorical features ({len(categorical_columns)}):", output_file)
    for col in categorical_columns:
        print_and_save(f"  - {col} ({df[col].dtype})", output_file)
    
    return numeric_columns, categorical_columns

def create_enhanced_preprocessing_pipeline(numeric_columns, categorical_columns):
    print_and_save("\nCreating enhanced preprocessing pipeline...", output_file)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ],
        remainder='drop'
    )
    
    return preprocessor

def preprocess_dataset(df, df_classification, dataset_name, output_file):
    print_and_save(f"\n{'='*60}", output_file)
    print_and_save(f"PREPROCESSING {dataset_name.upper()}", output_file)
    print_and_save(f"{'='*60}", output_file)
    
    original_shape = df.shape
    print_and_save(f"Original shape: {original_shape}", output_file)
    
    outcome_df = df_classification[df_classification['Category'] == 'Outcome']
    target_column = outcome_df['Column_Name'].iloc[0] if len(outcome_df) > 0 else None
    
    if not target_column or target_column not in df.columns:
        print_and_save(f"Warning: Target column '{target_column}' not found in dataset", output_file)
        return None, None, None
    
    y = df[target_column].copy()
    print_and_save(f"Target variable: {target_column}", output_file)
    print_and_save(f"Target distribution:", output_file)
    target_counts = y.value_counts(dropna=False)
    for value, count in target_counts.items():
        percentage = (count / len(y)) * 100
        print_and_save(f"  {value}: {count} ({percentage:.1f}%)", output_file)
    
    excluded_categories = ['Administrative (Excluded)', 'Outcome-Derived (Excluded)', 'Scoring System (Excluded)']
    excluded_df = df_classification[df_classification['Category'].isin(excluded_categories)]
    excluded_columns = excluded_df['Column_Name'].tolist()
    
    X = df.drop(columns=excluded_columns + [target_column], errors='ignore')
    print_and_save(f"After dropping excluded columns: {X.shape}", output_file)
    
    numeric_columns, categorical_columns = identify_column_types(X, df_classification, target_column)
    
    preprocessor = create_enhanced_preprocessing_pipeline(numeric_columns, categorical_columns)
    
    print_and_save("\nFitting preprocessing pipeline...", output_file)
    X_preprocessed = preprocessor.fit_transform(X)
    
    print_and_save(f"Preprocessed shape: {X_preprocessed.shape}", output_file)
    
    feature_names = []
    
    feature_names.extend([f"num_{col}" for col in numeric_columns])
    
    if categorical_columns:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        for i, col in enumerate(categorical_columns):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"cat_{col}_{cat}" for cat in categories])
    
    print_and_save(f"Total features after preprocessing: {len(feature_names)}", output_file)
    
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
    
    missing_after = X_preprocessed_df.isnull().sum().sum()
    print_and_save(f"Missing values after preprocessing: {missing_after}", output_file)
    
    return X_preprocessed_df, y, preprocessor

def main():
    global output_file
    output_file, txt_filename, pkl_filename = setup_output_files()
    
    print_and_save("DATA PREPROCESSING PIPELINE", output_file)
    print_and_save("="*60, output_file)
    print_and_save("Building comprehensive preprocessing pipeline", output_file)
    print_and_save("="*60, output_file)
    
    try:
        df_excel, df_csv, df_classification = load_datasets_and_classification()
        
        excel_X, excel_y, excel_preprocessor = preprocess_dataset(
            df_excel, df_classification, "Excel Dataset", output_file
        )
        
        csv_X, csv_y, csv_preprocessor = preprocess_dataset(
            df_csv, df_classification, "CSV Dataset", output_file
        )
        
        print_and_save(f"\n{'='*60}", output_file)
        print_and_save("SAVING PREPROCESSING RESULTS", output_file)
        print_and_save(f"{'='*60}", output_file)
        
        script_dir = Path(__file__).parent
        
        if excel_X is not None:
            excel_X_path = script_dir / "excel_preprocessed_features.csv"
            excel_y_path = script_dir / "excel_target.csv"
            excel_X.to_csv(excel_X_path, index=False)
            excel_y.to_csv(excel_y_path, index=False)
            print_and_save(f"Saved: {excel_X_path}", output_file)
            print_and_save(f"Saved: {excel_y_path}", output_file)
        
        if csv_X is not None:
            csv_X_path = script_dir / "csv_preprocessed_features.csv"
            csv_y_path = script_dir / "csv_target.csv"
            csv_X.to_csv(csv_X_path, index=False)
            csv_y.to_csv(csv_y_path, index=False)
            print_and_save(f"Saved: {csv_X_path}", output_file)
            print_and_save(f"Saved: {csv_y_path}", output_file)
        
        preprocessing_objects = {
            'excel_preprocessor': excel_preprocessor,
            'csv_preprocessor': csv_preprocessor,
            'excel_feature_names': excel_X.columns.tolist() if excel_X is not None else [],
            'csv_feature_names': csv_X.columns.tolist() if csv_X is not None else []
        }
        
        with open(pkl_filename, 'wb') as f:
            pickle.dump(preprocessing_objects, f)
        print_and_save(f"Saved: {pkl_filename}", output_file)
        
        print_and_save(f"\n{'='*60}", output_file)
        print_and_save("DATA PREPROCESSING COMPLETED", output_file)
        print_and_save(f"{'='*60}", output_file)
        
    except Exception as e:
        print_and_save(f"\nPREPROCESSING FAILED: {e}", output_file)
        import traceback
        print_and_save(traceback.format_exc(), output_file)
    
    output_file.close()
    print(f"\nReport saved to: {txt_filename}")

if __name__ == "__main__":
    main()