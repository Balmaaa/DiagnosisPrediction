import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import threading
import warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    """Load and manage trained models"""
    
    def __init__(self):
        self.models = {}
        self.pipeline = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessing pipeline"""
        
        base_path = Path(__file__).parent
        
        try:
            # Load preprocessing pipeline
            pipeline_path = base_path / "04_Preprocessing Pipeline" / "preprocessing_pipeline_20260202_083206.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)
                print("Preprocessing pipeline loaded successfully")
            
            # Load feature names from preprocessed data
            features_path = base_path / "04_Preprocessing Pipeline" / "excel_preprocessed_features.csv"
            if features_path.exists():
                features_df = pd.read_csv(features_path, nrows=1)
                self.feature_names = features_df.columns.tolist()
                print(f"Feature names loaded: {len(self.feature_names)} features")
            
            # Load Transformer Model
            transformer_path = base_path / "05_Transformer_Model"
            transformer_files = list(transformer_path.glob("transformer_model_*.pkl"))
            if transformer_files:
                latest_transformer = max(transformer_files, key=lambda x: x.stat().st_mtime)
                with open(latest_transformer, 'rb') as f:
                    self.models['Transformer'] = pickle.load(f)
                print(f"Transformer model loaded: {latest_transformer.name}")
            
            # Load Decision Tree Model
            dt_path = base_path / "06_Decision_Trees"
            dt_files = list(dt_path.glob("decision_tree_model_*.pkl"))
            if dt_files:
                latest_dt = max(dt_files, key=lambda x: x.stat().st_mtime)
                with open(latest_dt, 'rb') as f:
                    self.models['Decision Tree'] = pickle.load(f)
                print(f"Decision Tree model loaded: {latest_dt.name}")
            
            # Load Gradient Boosting Model
            gb_path = base_path / "07_Gradient_Boosting"
            gb_files = list(gb_path.glob("gradient_boosting_model_*.pkl"))
            if gb_files:
                latest_gb = max(gb_files, key=lambda x: x.stat().st_mtime)
                with open(latest_gb, 'rb') as f:
                    self.models['Gradient Boosting'] = pickle.load(f)
                print(f"Gradient Boosting model loaded: {latest_gb.name}")
            
            # Load XGBoost Model
            xgb_path = base_path / "08_XGBoost"
            xgb_files = list(xgb_path.glob("xgboost_model_*.pkl"))
            if xgb_files:
                latest_xgb = max(xgb_files, key=lambda x: x.stat().st_mtime)
                with open(latest_xgb, 'rb') as f:
                    self.models['XGBoost'] = pickle.load(f)
                print(f"XGBoost model loaded: {latest_xgb.name}")
                
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict(self, model_name, input_data):
        """Make prediction using selected model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing
        if self.pipeline:
            processed_data = self.pipeline.transform(df)
        else:
            processed_data = df
        
        # Make prediction
        if model_name == 'Transformer':
            # Special handling for Transformer
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
        else:
            # Scikit-learn models
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
        
        return prediction[0], prediction_proba[0]

class AppendicitisGUI:
    """Main GUI Application for Appendicitis Prediction"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System")
        self.root.geometry("1200x800")
        
        # Initialize model loader
        self.model_loader = ModelLoader()
        
        # Create main layout
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System", 
                                font=('Arial', 16, 'bold'))
        title_label.pack()
        
        # Model Selection
        model_frame = ttk.LabelFrame(self.root, text="Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.model_var = tk.StringVar()
        self.model_var.set("XGBoost")  # Default model
        
        for model_name in self.model_loader.models.keys():
            ttk.Radiobutton(model_frame, text=model_name, variable=self.model_var, 
                           value=model_name).pack(side=tk.LEFT, padx=10)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # Create tabs
        self.create_demographic_tab()
        self.create_clinical_tab()
        self.create_laboratory_tab()
        self.create_imaging_tab()
        
        # Prediction button and results
        self.create_prediction_section()
    
    def create_demographic_tab(self):
        """Create Demographic Information tab"""
        
        demo_frame = ttk.Frame(self.notebook)
        self.notebook.add(demo_frame, text="Demographic Information")
        
        # Create input fields
        self.demo_vars = {}
        
        fields = [
            ("Age (years):", "Age", 1, 0),
            ("Weight (kg):", "Weight", 1, 1),
            ("Height (cm):", "Height", 1, 2),
            ("BMI:", "BMI", 2, 0),
            ("Sex:", "Sex", 2, 1),
            ("Neutrophil Percentage (%):", "Neutrophil_Percentage", 2, 2)
        ]
        
        for label_text, field_name, row, col in fields:
            label = ttk.Label(demo_frame, text=label_text)
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            if field_name == "Sex":
                var = tk.StringVar(value="Male")
                combo = ttk.Combobox(demo_frame, textvariable=var, values=["Male", "Female"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            else:
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(demo_frame, textvariable=var)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            
            self.demo_vars[field_name] = var
    
    def create_clinical_tab(self):
        """Create Clinical Symptoms tab"""
        
        clinical_frame = ttk.Frame(self.notebook)
        self.notebook.add(clinical_frame, text="Clinical Symptoms")
        
        self.clinical_vars = {}
        
        # Create scrollable frame
        canvas = tk.Canvas(clinical_frame)
        scrollbar = ttk.Scrollbar(clinical_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        fields = [
            ("Body Temperature (°C):", "Body_Temperature", 0, 0, "float"),
            ("Lower Right Abdominal Pain:", "Lower_Right_Abd_Pain", 0, 1, "yesno"),
            ("Migratory Pain:", "Migratory_Pain", 0, 2, "yesno"),
            ("Loss of Appetite:", "Loss_of_Appetite", 1, 0, "yesno"),
            ("Nausea:", "Nausea", 1, 1, "yesno"),
            ("Coughing Pain:", "Coughing_Pain", 1, 2, "yesno"),
            ("Dysuria:", "Dysuria", 2, 0, "yesno"),
            ("Stool Changes:", "Stool", 2, 1, "yesno"),
            ("Peritonitis:", "Peritonitis", 2, 2, "yesno"),
            ("Severity:", "Severity", 3, 0, "severity"),
            ("Contralateral Rebound Tenderness:", "Contralateral_Rebound_Tenderness", 3, 1, "yesno"),
            ("Ipsilateral Rebound Tenderness:", "Ipsilateral_Rebound_Tenderness", 3, 2, "yesno"),
            ("Psoas Sign:", "Psoas_Sign", 4, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text)
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=36.5)
                entry = ttk.Entry(scrollable_frame, textvariable=var)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["yes", "no"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "severity":
                var = tk.StringVar(value="uncomplicated")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["uncomplicated", "complicated"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            
            self.clinical_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_laboratory_tab(self):
        """Create Laboratory Results tab"""
        
        lab_frame = ttk.Frame(self.notebook)
        self.notebook.add(lab_frame, text="Laboratory Results")
        
        self.lab_vars = {}
        
        fields = [
            ("WBC Count (×10^9/L):", "WBC_Count", 0, 0),
            ("RBC Count (×10^12/L):", "RBC_Count", 0, 1),
            ("Hemoglobin (g/dL):", "Hemoglobin", 0, 2),
            ("RDW (%):", "RDW", 1, 0),
            ("Segmented Neutrophils (%):", "Segmented_Neutrophils", 1, 1),
            ("Thrombocyte Count (×10^9/L):", "Thrombocyte_Count", 1, 2),
            ("CRP (mg/L):", "CRP", 2, 0),
            ("Neutrophilia:", "Neutrophilia", 2, 1, "yesno"),
            ("Ketones in Urine:", "Ketones_in_Urine", 2, 2, "ketones"),
            ("RBC in Urine:", "RBC_in_Urine", 3, 0, "yesno"),
            ("WBC in Urine:", "WBC_in_Urine", 3, 1, "yesno")
        ]
        
        for label_text, field_name, row, col, *extra in fields:
            label = ttk.Label(lab_frame, text=label_text)
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            if extra and extra[0] == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(lab_frame, textvariable=var, values=["yes", "no"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif extra and extra[0] == "ketones":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(lab_frame, textvariable=var, 
                                   values=["no", "+", "++", "+++"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            else:
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(lab_frame, textvariable=var)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            
            self.lab_vars[field_name] = var
    
    def create_imaging_tab(self):
        """Create Imaging Findings tab"""
        
        imaging_frame = ttk.Frame(self.notebook)
        self.notebook.add(imaging_frame, text="Imaging Findings")
        
        self.imaging_vars = {}
        
        # Create scrollable frame
        canvas = tk.Canvas(imaging_frame)
        scrollbar = ttk.Scrollbar(imaging_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        fields = [
            ("US Performed:", "US_Performed", 0, 0, "yesno"),
            ("US Number:", "US_Number", 0, 1, "int"),
            ("Appendix on US:", "Appendix_on_US", 0, 2, "yesno"),
            ("Appendix Diameter (mm):", "Appendix_Diameter", 1, 0, "float"),
            ("Appendix Wall Layers:", "Appendix_Wall_Layers", 1, 1, "wall"),
            ("Appendicolith:", "Appendicolith", 1, 2, "appendicolith"),
            ("Appendicular Abscess:", "Appendicular_Abscess", 2, 0, "abscess"),
            ("Bowel Wall Thickening:", "Bowel_Wall_Thickening", 2, 1, "yesno"),
            ("Conglomerate of Bowel Loops:", "Conglomerate_of_Bowel_Loops", 2, 2, "yesno"),
            ("Coprostasis:", "Coprostasis", 3, 0, "yesno"),
            ("Enteritis:", "Enteritis", 3, 1, "yesno"),
            ("Free Fluids:", "Free_Fluids", 3, 2, "yesno"),
            ("Ileus:", "Ileus", 4, 0, "yesno"),
            ("Meteorism:", "Meteorism", 4, 1, "yesno"),
            ("Perforation:", "Perforation", 4, 2, "yesno"),
            ("Perfusion:", "Perfusion", 5, 0, "yesno"),
            ("Target Sign:", "Target_Sign", 5, 1, "yesno"),
            ("Surrounding Tissue Reaction:", "Surrounding_Tissue_Reaction", 5, 2, "yesno"),
            ("Pathological Lymph Nodes:", "Pathological_Lymph_Nodes", 6, 0, "yesno"),
            ("Lymph Nodes Location:", "Lymph_Nodes_Location", 6, 1, "location"),
            ("Abscess Location:", "Abscess_Location", 6, 2, "abscess_location")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text)
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            if field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["yes", "no"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "int":
                var = tk.IntVar(value=0)
                entry = ttk.Entry(scrollable_frame, textvariable=var)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "float":
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(scrollable_frame, textvariable=var)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "wall":
                var = tk.StringVar(value="intact")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["intact", "partially raised", "raised", "upset"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "appendicolith":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["no", "suspected", "yes"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "abscess":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["no", "suspected", "yes"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "location":
                var = tk.StringVar(value="None")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["None", "Ileozökal", "Lymphadenopathie", "Mesenterial", 
                                          "Ovarialzysten", "UB", "Douglas", "an den M. psoas rechts", 
                                          "perityphlitisch", "re Mittelbauch", "rechter Mittelbauch", 
                                          "rechter Unterbauch", "retrovesikal"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "abscess_location":
                var = tk.StringVar(value="None")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["None", "Douglas", "an den M. psoas rechts", "perityphlitisch", 
                                          "re Mittelbauch", "rechter Mittelbauch", "rechter Unterbauch", 
                                          "retrovesikal"], state="readonly")
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            
            self.imaging_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_prediction_section(self):
        """Create prediction button and results section"""
        
        # Prediction button
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict Diagnosis", 
                                        command=self.predict_diagnosis, style="Accent.TButton")
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear Form", command=self.clear_form)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Results section
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def collect_input_data(self):
        """Collect all input data from the form"""
        
        input_data = {}
        
        # Collect demographic data
        for field_name, var in self.demo_vars.items():
            input_data[field_name] = var.get()
        
        # Collect clinical data
        for field_name, var in self.clinical_vars.items():
            input_data[field_name] = var.get()
        
        # Collect laboratory data
        for field_name, var in self.lab_vars.items():
            input_data[field_name] = var.get()
        
        # Collect imaging data
        for field_name, var in self.imaging_vars.items():
            input_data[field_name] = var.get()
        
        return input_data
    
    def validate_input(self, input_data):
        """Validate input data"""
        
        try:
            # Check required fields
            required_fields = ['Age', 'Weight', 'Height', 'Body_Temperature']
            
            for field in required_fields:
                if field in input_data:
                    value = input_data[field]
                    if isinstance(value, (int, float)) and value <= 0:
                        messagebox.showerror("Validation Error", f"{field} must be greater than 0")
                        return False
            
            # Check age range
            if 'Age' in input_data and input_data['Age'] > 18:
                messagebox.showwarning("Age Warning", "This system is designed for pediatric patients (≤18 years)")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Input validation failed: {e}")
            return False
    
    def predict_diagnosis(self):
        """Make prediction using selected model"""
        
        try:
            # Update status
            self.status_var.set("Processing prediction...")
            self.predict_button.config(state="disabled")
            
            # Collect input data
            input_data = self.collect_input_data()
            
            # Validate input
            if not self.validate_input(input_data):
                return
            
            # Make prediction in separate thread
            threading.Thread(target=self._make_prediction, args=(input_data,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_var.set("Error")
            self.predict_button.config(state="normal")
    
    def _make_prediction(self, input_data):
        """Internal method to make prediction"""
        
        try:
            # Get selected model
            model_name = self.model_var.get()
            
            # Make prediction
            prediction, prediction_proba = self.model_loader.predict(model_name, input_data)
            
            # Format results
            diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
            confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
            
            results = f"""
{'='*60}
PEDIATRIC APPENDICITIS PREDICTION RESULTS
{'='*60}

Model Used: {model_name}
Diagnosis: {diagnosis}
Confidence: {confidence:.2%}

Prediction Probabilities:
- Appendicitis: {prediction_proba[1]:.2%}
- No Appendicitis: {prediction_proba[0]:.2%}

Medical Interpretation:
{'='*60}
"""
            
            if prediction == 1:
                results += """
⚠️  HIGH RISK OF APPENDICITIS DETECTED
Recommendations:
- Immediate medical evaluation recommended
- Consider surgical consultation
- Monitor for worsening symptoms
- Prepare for possible appendectomy
"""
            else:
                results += """
✅ LOW RISK OF APPENDICITIS
Recommendations:
- Continue monitoring
- Consider alternative diagnoses
- Re-evaluate if symptoms worsen
- Follow-up examination recommended
"""
            
            results += f"""
{'='*60}
Note: This prediction is for medical reference only.
Clinical judgment should always prevail.
{'='*60}
"""
            
            # Update results display
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            self.root.after(0, self._display_error, error_msg)
    
    def _display_results(self, results):
        """Display prediction results"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.status_var.set("Prediction complete")
        self.predict_button.config(state="normal")
    
    def _display_error(self, error_msg):
        """Display error message"""
        
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("Error")
        self.predict_button.config(state="normal")
    
    def clear_form(self):
        """Clear all form fields"""
        
        # Clear demographic fields
        for var in self.demo_vars.values():
            if isinstance(var, tk.StringVar):
                var.set("")
            else:
                var.set(0.0)
        
        # Clear clinical fields
        for field_name, var in self.clinical_vars.items():
            if field_name == "Body_Temperature":
                var.set(36.5)
            elif field_name == "Severity":
                var.set("uncomplicated")
            else:
                var.set("no")
        
        # Clear laboratory fields
        for field_name, var in self.lab_vars.items():
            if isinstance(var, tk.StringVar):
                if field_name == "Ketones_in_Urine":
                    var.set("no")
                else:
                    var.set("no")
            else:
                var.set(0.0)
        
        # Clear imaging fields
        for field_name, var in self.imaging_vars.items():
            if isinstance(var, tk.StringVar):
                if field_name == "US_Number":
                    var.set("0")
                elif field_name == "Appendix_Wall_Layers":
                    var.set("intact")
                elif field_name in ["Appendicolith", "Appendicular_Abscess"]:
                    var.set("no")
                elif field_name in ["Lymph_Nodes_Location", "Abscess_Location"]:
                    var.set("None")
                else:
                    var.set("no")
            else:
                var.set(0)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Form cleared")

def main():
    """Main function to run the GUI application"""
    
    root = tk.Tk()
    app = AppendicitisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
