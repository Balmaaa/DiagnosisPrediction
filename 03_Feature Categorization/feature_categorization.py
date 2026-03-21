import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def setup_output():  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"feature_classification_{timestamp}.txt"
    csv_filename = f"feature_classification_{timestamp}.csv"
    
    script_dir = Path(__file__).parent
    txt_path = script_dir / txt_filename
    csv_path = script_dir / csv_filename
    
    txt_file = open(txt_path, 'w', encoding='utf-8')
    
    return txt_file, str(txt_path), str(csv_path)

def print_save(text, output_file):
    print(text)
    output_file.write(text + '\n')

def load_datasets():
    current_dir = Path.cwd()
    if current_dir.name == "03_Feature Categorization":
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

def classify_column(column_name):
    """
    Deterministically classify a column into semantic categories.
    
    Categories:
    - Outcome (Target)
    - Demographic
    - Clinical-Symptomatic
    - Laboratory
    - Imaging
    - Administrative (Excluded)
    - Outcome-Derived (Excluded)
    - Scoring System (Excluded)
    """
    
    col_lower = column_name.lower().strip()
    
    # OUTCOME (TARGET) - Highest priority
    outcome_keywords = ['diagnosis']
    if any(keyword in col_lower for keyword in outcome_keywords):
        return "Outcome"
    
    # OUTCOME-DERIVED (Excluded) - Check before Demographic to catch Management correctly
    outcome_derived_keywords = [
        'management', 'treatment', 'therapy', 'surgery', 'operation',
        'length_of_stay', 'hospital', 'discharge', 'admission',
        'complication', 'follow_up', 'recovery', 'intervention',
        'medication', 'procedure', 'duration', 'outcome', 'result',
        'prognosis', 'response', 'improvement', 'deterioration'
    ]
    if any(keyword in col_lower for keyword in outcome_derived_keywords):
        return "Outcome-Derived (Excluded)"
    
    # LABORATORY - Check before other categories for specific medical tests
    laboratory_keywords = [
        'blood', 'wbc', 'rbc', 'hemoglobin', 'hematocrit', 'platelet',
        'thrombocyte', 'crp', 'esr', 'inflammation', 'neutrophil',
        'neutrophil_percentage', 'lymphocyte', 'monocyte', 'basophil', 'eosinophil', 'rdw', 'mcv',
        'mch', 'mchc', 'urine', 'protein', 'glucose', 'electrolyte',
        'sodium', 'potassium', 'chloride', 'bicarbonate', 'creatinine',
        'bun', 'ast', 'alt', 'bilirubin', 'alkaline_phosphatase',
        'albumin', 'globulin', 'cholesterol', 'triglyceride', 'hba1c'
    ]
    if any(keyword in col_lower for keyword in laboratory_keywords):
        return "Laboratory"
    
    # DEMOGRAPHIC
    demographic_keywords = [
        'age', 'sex', 'gender', 'weight', 'height', 'bmi', 'body_mass',
        'ethnicity', 'race', 'nationality', 'insurance', 'patient_id'
    ]
    if any(keyword in col_lower for keyword in demographic_keywords):
        return "Demographic"
    
    # CLINICAL-SYMPTOMATIC - Check before imaging to catch clinical symptoms correctly
    clinical_keywords = [
        'pain', 'abdominal', 'abdomen', 'tenderness', 'rebound',
        'guarding', 'peritonitis', 'nausea', 'vomiting', 'vomit',
        'fever', 'temperature', 'appetite', 'diet', 'bowel', 'stool',
        'constipation', 'diarrhea', 'dysuria', 'frequency', 'urgency',
        'cough', 'migratory', 'location', 'onset', 'duration',
        'severity', 'intensity', 'character', 'radiation',
        'psoas_sign', 'rovsing_sign', 'obturator_sign',
        'clinical', 'symptom', 'sign', 'examination', 'physical',
        'history', 'present_illness', 'chief_complaint'
    ]
    if any(keyword in col_lower for keyword in clinical_keywords):
        return "Clinical-Symptomatic"
    
    # IMAGING
    imaging_keywords = [
        'us', 'ultrasound', 'sonography', 'ct', 'mri', 'x_ray', 'radiograph',
        'appendix_on_us', 'us_performed', 'us_number', 'appendix_diameter',
        'appendix_wall', 'wall_layers', 'target_sign', 'free_fluids',
        'perfusion', 'appendicolith', 'coprostasis', 'perforation',
        'abscess', 'bowel_wall', 'lymph_nodes', 'surrounding_tissue',
        'meteorism', 'ileus', 'enteritis', 'conglomerate'
    ]
    if any(keyword in col_lower for keyword in imaging_keywords):
        return "Imaging"
    
    # SCORING SYSTEM (Excluded)
    scoring_keywords = [
        'score', 'scale', 'index', 'alvarado', 'pas', 'pediatric',
        'appendicitis_score', 'risk', 'probability', 'likelihood',
        'assessment', 'rating', 'grade'
    ]
    if any(keyword in col_lower for keyword in scoring_keywords):
        return "Scoring System (Excluded)"
    
    # ADMINISTRATIVE (Excluded)
    administrative_keywords = [
        'id', 'number', 'date', 'time', 'hour', 'minute', 'second',
        'record', 'file', 'chart', 'department', 'ward', 'room',
        'bed', 'physician', 'doctor', 'nurse', 'staff', 'clinic',
        'hospital', 'center', 'facility', 'insurance', 'billing',
        'cost', 'charge', 'payment', 'authorization', 'referral',
        'transfer', 'consultation', 'appointment', 'schedule',
        'protocol', 'guideline', 'policy', 'form', 'document',
        'unnamed', 'index', 'row', 'column'
    ]
    if any(keyword in col_lower for keyword in administrative_keywords):
        return "Administrative (Excluded)"
    
    # GYNECOLOGICAL/OTHER SPECIALTY FINDINGS (Excluded)
    specialty_keywords = ['gynecological', 'obstetric', 'urological', 'orthopedic', 'cardiac', 'pulmonary', 'neurological', 'dermatological', 'finding', 'findings', 'examination', 'test', 'result']
    if any(keyword in col_lower for keyword in specialty_keywords):
        return "Administrative (Excluded)"
    
    raise ValueError(f"Unable to classify column: '{column_name}'")

def classify_dataset_columns(df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"CLASSIFYING COLUMNS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    classifications = {}
    unclassified_columns = []
    
    for col in df.columns:
        try:
            category = classify_column(col)
            classifications[col] = category
            print_save(f"  {col:<40} -> {category}", output_file)
        except ValueError as e:
            unclassified_columns.append(col)
            print_save(f"  {col:<40} -> UNCLASSIFIED (ERROR)", output_file)
    
    if unclassified_columns:
        error_msg = f"ERROR: {len(unclassified_columns)} columns could not be classified in {dataset_name}"
        print_save(f"\n{error_msg}", output_file)
        for col in unclassified_columns:
            print_save(f"  - {col}", output_file)
        raise ValueError(error_msg)
    
    print_save(f"\nSuccessfully classified all {len(classifications)} columns in {dataset_name}", output_file)
    return classifications

def classification_report(excel_classifications, csv_classifications, csv_filename):
    all_classifications = []
    
    for col, category in excel_classifications.items():
        all_classifications.append({'Dataset': 'Excel', 'Column_Name': col, 'Category': category})
    
    for col, category in csv_classifications.items():
        all_classifications.append({'Dataset': 'CSV', 'Column_Name': col, 'Category': category})
    
    df_report = pd.DataFrame(all_classifications)
    df_report = df_report.sort_values(['Dataset', 'Column_Name'])
    df_report.to_csv(csv_filename, index=False, encoding='utf-8')
    
    return df_report

def classification_summary(excel_classifications, csv_classifications, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save("CLASSIFICATION SUMMARY", output_file)
    print_save(f"{'='*60}", output_file)
    
    def count_categories(classifications, dataset_name):
        categories = {}
        for category in classifications.values():
            categories[category] = categories.get(category, 0) + 1
        
        print_save(f"\n{dataset_name}:", output_file)
        for category, count in sorted(categories.items()):
            print_save(f"  {category:<30} : {count:>3} columns", output_file)
        print_save(f"  {'Total':<30} : {len(classifications):>3} columns", output_file)
        return categories
    
    excel_counts = count_categories(excel_classifications, "Excel Dataset")
    csv_counts = count_categories(csv_classifications, "CSV Dataset")
    
    print_save(f"\nOverall Summary:", output_file)
    all_categories = set(excel_counts.keys()) | set(csv_counts.keys())
    for category in sorted(all_categories):
        total = excel_counts.get(category, 0) + csv_counts.get(category, 0)
        print_save(f"  {category:<30} : {total:>3} columns", output_file)
    
    total_columns = len(excel_classifications) + len(csv_classifications)
    print_save(f"  {'Total':<30} : {total_columns:>3} columns", output_file)

def main():
    global output_file
    output_file, txt_filename, csv_filename = setup_output()
    
    print_save("FEATURE SEMANTIC CATEGORIZATION", output_file)
    print_save("="*60, output_file)
    print_save("Classifying all columns into semantic categories", output_file)
    print_save("="*60, output_file)
    
    df_excel, df_csv = load_datasets()
    
    if df_excel is None and df_csv is None:
        print_save("[ERROR] No datasets could be loaded. Exiting.", output_file)
        output_file.close()
        print(f"\nReport saved to: {txt_filename}")
        return
    
    try:
        excel_classifications = {}
        if df_excel is not None:
            excel_classifications = classify_dataset_columns(df_excel, "Excel Dataset", output_file)
        
        csv_classifications = {}
        if df_csv is not None:
            csv_classifications = classify_dataset_columns(df_csv, "CSV Dataset", output_file)
        
        df_report = classification_report(excel_classifications, csv_classifications, csv_filename)
        print_save(f"\nCSV classification report saved to: {csv_filename}", output_file)
        
        classification_summary(excel_classifications, csv_classifications, output_file)
        
        print_save(f"\n{'='*60}", output_file)
        print_save("FEATURE SEMANTIC CATEGORIZATION COMPLETED", output_file)
        print_save(f"{'='*60}", output_file)
        
    except ValueError as e:
        print_save(f"\nCLASSIFICATION FAILED: {e}", output_file)
        print_save("Please review the classification logic and update keywords.", output_file)
    
    output_file.close()
    print(f"\nText report saved to: {txt_filename}")

if __name__ == "__main__":
    main()