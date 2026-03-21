import pandas as pd
from pathlib import Path
from datetime import datetime

def setup_output():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"target_leakage_analysis_{timestamp}.txt"
    
    script_dir = Path(__file__).parent
    output_path = script_dir / output_filename
    output_file = open(output_path, 'w', encoding='utf-8')
    
    return output_file, str(output_path)

def print_save(text, output_file):
    print(text)
    output_file.write(text + '\n')

def load_datasets():
    current_dir = Path.cwd()
    if current_dir.name == "02_Target Leak Control":
        excel_path = "../Dataset 1/app_data.xlsx"
        csv_path = "../Dataset 2/Regensburg Pediatric Appendicitis.csv"
    else:
        excel_path = "Dataset 1/app_data.xlsx"
        csv_path = "Dataset 2/Regensburg Pediatric Appendicitis.csv"
    
    try:
        df_excel = pd.read_excel(excel_path)
        print_save(f"[OK] Loaded Excel dataset: {excel_path}", output_file)
    except Exception as e:
        print_save(f"[ERROR] Error loading Excel dataset: {e}", output_file)
        return None, None
    
    try:
        df_csv = pd.read_csv(csv_path)
        print_save(f"[OK] Loaded CSV dataset: {csv_path}", output_file)
    except Exception as e:
        print_save(f"[ERROR] Error loading CSV dataset: {e}", output_file)
        return df_excel, None
    
    return df_excel, df_csv

def value_distribution(df, column_name):
    if column_name not in df.columns:
        return None
    
    series = df[column_name]
    value_counts = series.value_counts(dropna=False)
    unique_count = series.nunique()
    missing_count = series.isnull().sum()
    total_count = len(series)
    
    return {
        'value_counts': value_counts,
        'unique_count': unique_count,
        'missing_count': missing_count,
        'total_count': total_count,
        'missing_percentage': (missing_count / total_count) * 100
    }

def diagnosis_columns(df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"DIAGNOSIS COLUMN ANALYSIS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    diagnosis_keywords = ['diagnosis', 'diagnose', 'final', 'result', 'outcome', 'conclusion', 'finding', 'assessment', 'determination']
    
    potential_diagnosis_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in diagnosis_keywords):
            potential_diagnosis_cols.append(col)
    
    print_save(f"Potential diagnosis columns found: {len(potential_diagnosis_cols)}", output_file)
    
    for col in potential_diagnosis_cols:
        print_save(f"\nAnalyzing column: '{col}'", output_file)
        dist = value_distribution(df, col)
        
        if dist:
            print_save(f"  Unique values: {dist['unique_count']}", output_file)
            print_save(f"  Missing: {dist['missing_count']} ({dist['missing_percentage']:.2f}%)", output_file)
            print_save(f"  Value distribution:", output_file)
            
            for i, (value, count) in enumerate(dist['value_counts'].head(10).items()):
                percentage = (count / dist['total_count']) * 100
                value_str = str(value) if pd.notna(value) else 'NaN'
                print_save(f"    {value_str}: {count} ({percentage:.1f}%)", output_file)
            
            if len(dist['value_counts']) > 10:
                print_save(f"    ... and {len(dist['value_counts']) - 10} more values", output_file)
    
    return potential_diagnosis_cols

def exclude_presumptive_diagnoses(columns, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"EXCLUDING PRESUMPTIVE DIAGNOSES - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    presumptive_keywords = ['presumptive', 'preliminary', 'initial', 'suspected', 'provisional', 'working', 'tentative', 'probable', 'possible', 'differential']
    
    excluded_cols = []
    remaining_cols = []
    
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in presumptive_keywords):
            excluded_cols.append(col)
            print_save(f"EXCLUDED: '{col}' - Contains presumptive keyword", output_file)
        else:
            remaining_cols.append(col)
    
    print_save(f"\nPresumptive diagnoses excluded: {len(excluded_cols)}", output_file)
    print_save(f"Remaining diagnosis columns: {len(remaining_cols)}", output_file)
    
    return remaining_cols, excluded_cols

def exclude_outcome_derived_variables(columns, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"EXCLUDING OUTCOME-DERIVED VARIABLES - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    outcome_keywords = ['management', 'treatment', 'therapy', 'surgery', 'operation', 'length_of_stay', 'hospital', 'discharge', 'admission', 'complication', 'follow_up', 'recovery', 'intervention', 'medication', 'procedure', 'duration']
    
    excluded_cols = []
    remaining_cols = []
    
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in outcome_keywords):
            excluded_cols.append(col)
            print_save(f"EXCLUDED: '{col}' - Contains outcome-derived keyword", output_file)
        else:
            remaining_cols.append(col)
    
    print_save(f"\nOutcome-derived variables excluded: {len(excluded_cols)}", output_file)
    print_save(f"Remaining diagnosis columns: {len(remaining_cols)}", output_file)
    
    return remaining_cols, excluded_cols

def exclude_scoring_systems(columns, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"EXCLUDING SCORING SYSTEMS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    scoring_keywords = ['score', 'scale', 'index', 'alvarado', 'pas', 'pediatric', 'appendicitis_score', 'risk', 'probability', 'likelihood']
    
    excluded_cols = []
    remaining_cols = []
    
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in scoring_keywords):
            excluded_cols.append(col)
            print_save(f"EXCLUDED: '{col}' - Contains scoring system keyword", output_file)
        else:
            remaining_cols.append(col)
    
    print_save(f"\nScoring systems excluded: {len(excluded_cols)}", output_file)
    print_save(f"Remaining diagnosis columns: {len(remaining_cols)}", output_file)
    
    return remaining_cols, excluded_cols

def exclude_high_missing_and_specialty_columns(columns, df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"EXCLUDING HIGH MISSING & SPECIALTY COLUMNS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    specialty_keywords = ['gynecological', 'obstetric', 'urological', 'orthopedic', 'cardiac', 'pulmonary', 'neurological', 'dermatological', 'finding', 'findings', 'examination', 'test', 'result']
    
    excluded_cols = []
    remaining_cols = []
    
    for col in columns:
        col_lower = col.lower()
        exclude_reason = None
        
        if col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 80:
                exclude_reason = f"High missing values ({missing_pct:.1f}%)"
        
        if exclude_reason is None and any(keyword in col_lower for keyword in specialty_keywords):
            exclude_reason = "Specialty-specific finding, not final diagnosis"
        
        if exclude_reason:
            excluded_cols.append(col)
            print_save(f"EXCLUDED: '{col}' - {exclude_reason}", output_file)
        else:
            remaining_cols.append(col)
    
    print_save(f"\nHigh missing/specialty columns excluded: {len(excluded_cols)}", output_file)
    print_save(f"Remaining diagnosis columns: {len(remaining_cols)}", output_file)
    
    return remaining_cols, excluded_cols

def validate_single_target(final_columns, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"TARGET VALIDATION - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    if len(final_columns) == 0:
        error_msg = f"ERROR: No valid target columns found in {dataset_name}"
        print_save(error_msg, output_file)
        raise ValueError(error_msg)
    elif len(final_columns) == 1:
        target_col = final_columns[0]
        print_save(f"SUCCESS: Exactly one target column identified: '{target_col}'", output_file)
        return target_col
    else:
        error_msg = f"ERROR: {len(final_columns)} target columns found in {dataset_name}. Expected exactly 1."
        print_save(error_msg, output_file)
        print_save("Remaining columns:", output_file)
        for i, col in enumerate(final_columns, 1):
            print_save(f"  {i}. {col}", output_file)
        raise ValueError(error_msg)

def main():
    global output_file
    output_file, output_filename = setup_output()
    
    print_save("TARGET & LEAKAGE CONTROL ANALYSIS", output_file)
    print_save("="*60, output_file)
    print_save("Identifying final diagnosis column while excluding leakage variables", output_file)
    print_save("="*60, output_file)
    
    df_excel, df_csv = load_datasets()
    
    if df_excel is None and df_csv is None:
        print_save("[ERROR] No datasets could be loaded. Exiting.", output_file)
        output_file.close()
        print(f"\nReport saved to: {output_filename}")
        return
    
    final_targets = {}
    
    if df_excel is not None:
        potential_diags = diagnosis_columns(df_excel, "Excel Dataset", output_file)                                # Identifying potential diagnosis columns
        
        if potential_diags:
            remaining1, excluded_presump = exclude_presumptive_diagnoses(potential_diags, "Excel Dataset", output_file)     # Exclude presumptive diagnoses
            remaining2, excluded_outcome = exclude_outcome_derived_variables(remaining1, "Excel Dataset", output_file)      # Exclude outcome-derived variables
            remaining3, excluded_scoring = exclude_scoring_systems(remaining2, "Excel Dataset", output_file)                # Exclude scoring systmes (ALVARADO & RIPASA)
            remaining4, excluded_high_missing = exclude_high_missing_and_specialty_columns(remaining3, df_excel, "Excel Dataset", output_file)      # Exclude high missing and specialty columns
            
            if remaining4:                                                                                                  # Data Validation between single targets
                try:
                    target_col = validate_single_target(remaining4, "Excel Dataset", output_file)
                    final_targets['Excel'] = target_col
                except ValueError as e:
                    print_save(f"Validation failed for Excel dataset: {e}", output_file)
            else:
                print_save("No columns remaining after exclusions for Excel dataset", output_file)
        else:
            print_save("No potential diagnosis columns found in Excel dataset", output_file)
    
    if df_csv is not None:
        potential_diags = diagnosis_columns(df_csv, "CSV Dataset", output_file)                                    # Identifying potential diagnosis columns
        
        if potential_diags:
            remaining1, excluded_presump = exclude_presumptive_diagnoses(potential_diags, "CSV Dataset", output_file)       # Exclude presumptive diagnoses
            remaining2, excluded_outcome = exclude_outcome_derived_variables(remaining1, "CSV Dataset", output_file)        # Exclude outcome-derived variables
            remaining3, excluded_scoring = exclude_scoring_systems(remaining2, "CSV Dataset", output_file)                  # Exclude scoring systmes (ALVARADO & RIPASA)
            remaining4, excluded_high_missing = exclude_high_missing_and_specialty_columns(remaining3, df_csv, "CSV Dataset", output_file)      # Exclude high missing and specialty columns
            
            if remaining4:                                                                                                  # Data Validation between single targets
                try:
                    target_col = validate_single_target(remaining4, "CSV Dataset", output_file)
                    final_targets['CSV'] = target_col
                except ValueError as e:
                    print_save(f"Validation failed for CSV dataset: {e}", output_file)
            else:
                print_save("No columns remaining after exclusions for CSV dataset", output_file)
        else:
            print_save("No potential diagnosis columns found in CSV dataset", output_file)
    
    print_save(f"\n{'='*60}", output_file)
    print_save("FINAL TARGET SUMMARY", output_file)
    print_save(f"{'='*60}", output_file)
    
    for dataset, target in final_targets.items():
        print_save(f"{dataset} Dataset Target: '{target}'", output_file)
    
    if not final_targets:
        print_save("WARNING: No valid targets identified in any dataset", output_file)
    
    print_save(f"\n{'='*60}", output_file)
    print_save("TARGET & LEAKAGE CONTROL ANALYSIS COMPLETED", output_file)
    print_save(f"{'='*60}", output_file)
    
    output_file.close()
    print(f"\nReport saved to: {output_filename}")

if __name__ == "__main__":
    main()