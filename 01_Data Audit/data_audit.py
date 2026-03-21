import pandas as pd
from pathlib import Path
import re
from datetime import datetime

def normalize_col(name):
    norm = str(name).lower()                    # Convert content values to lowercase
    norm = re.sub(r'[^a-z0-9_]', '_', norm)     # Replace special characters with underscores
    norm = re.sub(r'_+', '_', norm)             # Replace multiple underscores with single underscore
    norm = norm.strip('_')                      # Remove leading/trailing underscores
    return norm

def setup_output_file():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data_audit_report_{timestamp}.txt"
    
    script_dir = Path(__file__).parent
    output_path = script_dir / output_file
    output_file = open(output_path, 'w', encoding='utf-8')
    
    return output_file, str(output_path)

def print_save(text, output_file):
    print(text)
    output_file.write(text + '\n')

def load_datasets(output_file):  
    current_dir = Path.cwd()
    if current_dir.name == "01_Data Audit":
        excel_path = "../Dataset 1/app_data.xlsx"
        csv_path = "../Dataset 2/Regensburg Pediatric Appendicitis.csv"
    else:
        excel_path = "Dataset 1/app_data.xlsx"
        csv_path = "Dataset 2/Regensburg Pediatric Appendicitis.csv"
    
    try:
        df_excel = pd.read_excel(excel_path)
        print_save(f"[OK] Successfully loaded Excel dataset: {excel_path}", output_file)
    except Exception as e:
        print_save(f"[ERROR] Error loading Excel dataset: {e}", output_file)
        return None, None, output_file
    
    try:
        df_csv = pd.read_csv(csv_path)
        print_save(f"[OK] Successfully loaded CSV dataset: {csv_path}", output_file)
    except Exception as e:
        print_save(f"[ERROR] Error loading CSV dataset: {e}", output_file)
        return df_excel, None, output_file
    
    return df_excel, df_csv, output_file

def display_info(df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"DATASET: {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    print_save(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns", output_file)       # Display data shpae
    
    print_save(f"\nColumn Names ({len(df.columns)} total):", output_file)               # Display column names
    for i, col in enumerate(df.columns, 1):
        print_save(f"  {i:2d}. {col}", output_file)
    
    print_save(f"\nData Types:", output_file)                                           # Display data types
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print_save(f"  {dtype}: {count} columns", output_file)
    
    print_save(f"\nDetailed Data Types:", output_file)
    for col, dtype in df.dtypes.items():
        print_save(f"  {col}: {dtype}", output_file)

def analyze_columns(df1, df2, name1, name2, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save("COLUMN COMPARISON", output_file)
    print_save(f"{'='*60}", output_file)
    
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1.intersection(cols2)
    print_save(f"\nCommon Columns ({len(common_cols)}):", output_file)
    if common_cols:
        for i, col in enumerate(sorted(common_cols), 1):
            print_save(f"  {i}. {col}", output_file)
    else:
        print_save("  No common columns found", output_file)
    
    unique1 = cols1 - cols2
    print_save(f"\nColumns Unique to {name1} ({len(unique1)}):", output_file)
    if unique1:
        for i, col in enumerate(sorted(unique1), 1):
            print_save(f"  {i}. {col}", output_file)
    else:
        print_save("  No unique columns", output_file)
    
    unique2 = cols2 - cols1
    print_save(f"\nColumns Unique to {name2} ({len(unique2)}):", output_file)
    if unique2:
        for i, col in enumerate(sorted(unique2), 1):
            print_save(f"  {i}. {col}", output_file)
    else:
        print_save("  No unique columns", output_file)

def analysis_normalize_col(df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"COLUMN NAME NORMALIZATION ANALYSIS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    print_save(f"{'Original Name':<40} {'norm Name':<40}", output_file)
    print_save(f"{'-'*40} {'-'*40}", output_file)
    
    for col in df.columns:
        norm = normalize_col(col)
        print_save(f"{col:<40} {norm:<40}", output_file)

def analysis_missing_value(df, dataset_name, output_file):
    print_save(f"\n{'='*60}", output_file)
    print_save(f"MISSING VALUES ANALYSIS - {dataset_name}", output_file)
    print_save(f"{'='*60}", output_file)
    
    missing_counts = df.isnull().sum()                          # Display percentage of missing values
    missing_percentages = (missing_counts / len(df)) * 100
    
    total_missing = missing_counts.sum()                        # Display summary of all missing value percentages
    total_cells = df.shape[0] * df.shape[1]
    overall_missing_percentage = (total_missing / total_cells) * 100
    
    print_save(f"Overall Missing Data: {total_missing:,} cells ({overall_missing_percentage:.2f}% of all data)", output_file)
    print_save(f"Columns with missing data: {(missing_percentages > 0).sum()} out of {len(df.columns)}", output_file)
    
    print_save(f"\nMissing Values by Column:", output_file)
    print_save(f"{'Column Name':<40} {'Missing Count':<15} {'Missing %':<10} {'Status':<15}", output_file)
    print_save(f"{'-'*40} {'-'*15} {'-'*10} {'-'*15}", output_file)
    
    missing_data = pd.DataFrame({'Column': df.columns, 'Missing Count': missing_counts.values, 'Missing %': missing_percentages.values})
    missing_data = missing_data.sort_values('Missing %', ascending=False)
    
    for _, row in missing_data.iterrows():                      # Percentage of missing data are based on online sources
        col = row['Column']                             
        count = row['Missing Count']
        pct = row['Missing %']
        
        if count == 0:
            status = "Complete"
        elif pct < 5:
            status = "Good (<5%)"
        elif pct < 20:
            status = "Fair (5-20%)"
        elif pct < 50:
            status = "Poor (20-50%)"
        else:
            status = "Critical (>50%)"
        
        print_save(f"{col:<40} {count:<15,} {pct:<10.2f}% {status:<15}", output_file)

def main():
    output_file, output_file = setup_output_file()
    
    print_save("DATA AUDIT REPORT", output_file)
    print_save("="*60, output_file)
    print_save("Analyzing two datasets: Excel and CSV files", output_file)
    print_save("="*60, output_file)
    
    df_excel, df_csv, output_file = load_datasets(output_file)
    
    if df_excel is None and df_csv is None:
        print_save("[ERROR] No datasets could be loaded. Exiting.", output_file)
        output_file.close()
        print(f"\nReport saved to: {output_file}")
        return
    
    if df_excel is not None:
        display_info(df_excel, "Excel Dataset (app_data.xlsx)", output_file)
    
    if df_csv is not None:
        display_info(df_csv, "CSV Dataset (Regensburg Pediatric Appendicitis.csv)", output_file)
    
    if df_excel is not None and df_csv is not None:
        analyze_columns(df_excel, df_csv, "Excel Dataset", "CSV Dataset", output_file)
    
    if df_excel is not None:
        analysis_normalize_col(df_excel, "Excel Dataset", output_file)
    
    if df_csv is not None:
        analysis_normalize_col(df_csv, "CSV Dataset", output_file)
    
    if df_excel is not None:
        analysis_missing_value(df_excel, "Excel Dataset", output_file)
    
    if df_csv is not None:
        analysis_missing_value(df_csv, "CSV Dataset", output_file)
    
    print_save(f"\n{'='*60}", output_file)
    print_save("DATA AUDIT COMPLETED", output_file)
    print_save(f"{'='*60}", output_file)
    
    output_file.close()
    print(f"\nReport saved to: {output_file}")

if __name__ == "__main__":
    main()