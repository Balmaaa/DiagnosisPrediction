import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import warnings
warnings.filterwarnings('ignore')

# Import the real backend predictor
from backend_predictor import AppendicitisPredictor

class PrecisionFocusedGUI:
    """GUI focused on precision and specificity with real AI models"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System - Real AI Models")
        self.root.geometry("2000x900")
        
        # Initialize real AI predictor
        self.predictor = AppendicitisPredictor()
        
        # Create main layout
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI with left input and right output"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System", 
                                font=('Arial', 20, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Precision & Specificity Focused - Right-Side Output", 
                                  font=('Arial', 12, 'italic'))
        subtitle_label.pack()
        
        # Model Selection with Real AI Models
        model_frame = ttk.LabelFrame(self.root, text="Real AI Model Selection - Precision & Specificity Focused", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="Transformer")  # Default to best precision/specificity
        
        # Get available models from real predictor
        available_models = self.predictor.get_available_models()
        
        for i, model in enumerate(available_models):
            model_info = self.predictor.get_model_info(model)
            model_text = f"{model}\n({model_info['type']})\nTrained: {model_info['trained']}"
            ttk.Radiobutton(model_frame, text=model_text, variable=self.model_var, 
                           value=model).grid(row=0, column=i, padx=15, sticky=tk.W)
        
        # Main container: Left (input) + Right (output)
        main_container = ttk.Frame(self.root)
        main_container.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)
        
        # LEFT SIDE: Input Form
        self.create_input_form(main_container)
        
        # RIGHT SIDE: Output Results
        self.create_output_section(main_container)
        
        # Bottom: Control Buttons
        self.create_control_section()
        
    def create_input_form(self, parent):
        """Create left-side input form"""
        
        # Input container
        input_frame = ttk.LabelFrame(parent, text="📋 PATIENT INPUT DATA", padding="15")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Create scrollable form
        canvas = tk.Canvas(input_frame, width=800)
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # SEGMENT 1: DEMOGRAPHIC INFORMATION
        self.create_demographic_segment(scrollable_frame)
        
        # SEGMENT 2: CLINICAL SYMPTOMS  
        self.create_clinical_segment(scrollable_frame)
        
        # SEGMENT 3: LABORATORY RESULTS
        self.create_laboratory_segment(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_output_section(self, parent):
        """Create right-side output section"""
        
        # Output container
        output_frame = ttk.LabelFrame(parent, text="📊 PREDICTION RESULTS & ANALYSIS", padding="15")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(output_frame, height=35, width=80, 
                                                font=('Courier', 10), wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial welcome message
        welcome_msg = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                      PRECISION & SPECIFICITY FOCUSED APPENDICITIS PREDICTION SYSTEM                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                      ║
║  🎯 STUDY FOCUS: PRECISION & SPECIFICITY                                                                         ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                                                                      ║
║  📊 KEY METRICS:                                                                                                 ║
║  • Precision: TP/(TP+FP) - How many predicted positives are actually appendicitis                                        ║
║  • Specificity: TN/(TN+FP) - How many non-appendicitis cases are correctly identified                                   ║
║  • False Positive Rate: FP/(FP+TN) - How often healthy patients are incorrectly flagged                               ║
║  • False Negative Rate: FN/(FN+TP) - How often appendicitis cases are missed                                       ║
║                                                                                                                      ║
║  🏆 BEST MODELS FOR PRECISION/SPECIFICITY:                                                                        ║
║  • Transformer: Highest Precision (75.6%) & Specificity (79.5%)                                                    ║
║  • Decision Tree: High Specificity (76.4%) & Good Precision (73.9%)                                                 ║
║  • Gradient Boosting: Balanced Precision (74.3%) & Specificity (71.7%)                                              ║
║  • XGBoost: Balanced Performance (75.1% Precision, 71.7% Specificity)                                            ║
║                                                                                                                      ║
║  🎯 CLINICAL APPLICATION:                                                                                          ║
║  • High Precision: Fewer false positives - reduces unnecessary surgeries                                                ║
║  • High Specificity: Fewer false negatives - reduces missed appendicitis cases                                           ║
║  • Optimal for: Emergency departments where both false positives and false negatives have clinical consequences              ║
║                                                                                                                      ║
║  📋 INSTRUCTIONS:                                                                                                 ║
║  1. Enter patient data in the form on the left                                                                       ║
║  2. Select model based on precision/specificity needs                                                               ║
║  3. Click "Predict Diagnosis" for detailed analysis                                                                   ║
║  4. Review precision-specificity metrics in results                                                                    ║
║                                                                                                                      ║
║  ⚕️  MEDICAL DISCLAIMER:                                                                                           ║
║  This system is for medical reference and decision support only. Clinical judgment should always prevail in patient care.       ║
║                                                                                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

"""
        self.results_text.insert(1.0, welcome_msg)
        
    def create_control_section(self):
        """Create bottom control section"""
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        # Style buttons
        style = ttk.Style()
        style.configure("Predict.TButton", font=('Arial', 14, 'bold'), padding=15)
        style.configure("Clear.TButton", font=('Arial', 11), padding=10)
        style.configure("Test.TButton", font=('Arial', 10), padding=8)
        
        self.predict_button = ttk.Button(button_frame, text="🔍 Predict Diagnosis", 
                                        command=self.predict_diagnosis, style="Predict.TButton")
        self.predict_button.pack(side=tk.LEFT, padx=20)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="🧹 Clear Form", command=self.clear_form, style="Clear.TButton")
        clear_button.pack(side=tk.LEFT, padx=15)
        
        # Quick fill buttons for testing
        test_frame = ttk.Frame(button_frame)
        test_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(test_frame, text="Quick Test Cases:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=10)
        ttk.Button(test_frame, text="High Precision Test", command=self.fill_precision_test, style="Test.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(test_frame, text="High Specificity Test", command=self.fill_specificity_test, style="Test.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(test_frame, text="Balanced Test", command=self.fill_balanced_test, style="Test.TButton").pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Precision & Specificity Focused System")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, font=('Arial', 11))
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
    
    def create_demographic_segment(self, parent):
        """Create demographic information segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="📋 SEGMENT 1: DEMOGRAPHIC INFORMATION", padding="12")
        segment_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.demo_vars = {}
        
        # Demographic fields in 2 columns
        fields = [
            ("Age (years):", "Age", 0, 0, "float"),
            ("Weight (kg):", "Weight", 0, 1, "float"),
            ("Height (cm):", "Height", 1, 0, "float"),
            ("BMI:", "BMI", 1, 1, "float"),
            ("Sex:", "Sex", 2, 0, "sex"),
            ("Neutrophil %:", "Neutrophil_Percentage", 2, 1, "float")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=12, pady=8, sticky=tk.W)
            
            if field_type == "sex":
                var = tk.StringVar(value="Male")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["Male", "Female"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=8)
            else:
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(segment_frame, textvariable=var, width=16, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=12, pady=8)
            
            self.demo_vars[field_name] = var
    
    def create_clinical_segment(self, parent):
        """Create clinical symptoms segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="🏥 SEGMENT 2: CLINICAL SYMPTOMS", padding="12")
        segment_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.clinical_vars = {}
        
        # Clinical symptoms in 2 columns
        fields = [
            ("Body Temp (°C):", "Body_Temperature", 0, 0, "float"),
            ("Lower Right Pain:", "Lower_Right_Abd_Pain", 0, 1, "yesno"),
            ("Migratory Pain:", "Migratory_Pain", 1, 0, "yesno"),
            ("Loss of Appetite:", "Loss_of_Appetite", 1, 1, "yesno"),
            ("Nausea:", "Nausea", 2, 0, "yesno"),
            ("Coughing Pain:", "Coughing_Pain", 2, 1, "yesno"),
            ("Dysuria:", "Dysuria", 3, 0, "yesno"),
            ("Stool Changes:", "Stool", 3, 1, "stool"),
            ("Peritonitis:", "Peritonitis", 4, 0, "yesno"),
            ("Severity:", "Severity", 4, 1, "severity"),
            ("Contralateral Rebound:", "Contralateral_Rebound_Tenderness", 5, 0, "yesno"),
            ("Ipsilateral Rebound:", "Ipsilateral_Rebound_Tenderness", 5, 1, "yesno"),
            ("Psoas Sign:", "Psoas_Sign", 6, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=12, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=36.5)
                entry = ttk.Entry(segment_frame, textvariable=var, width=16, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=12, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=6)
            elif field_type == "severity":
                var = tk.StringVar(value="uncomplicated")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["uncomplicated", "complicated"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=6)
            elif field_type == "stool":
                var = tk.StringVar(value="normal")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["normal", "constipation", "diarrhea"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=6)
            
            self.clinical_vars[field_name] = var
    
    def create_laboratory_segment(self, parent):
        """Create laboratory results segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="🔬 SEGMENT 3: LABORATORY RESULTS", padding="12")
        segment_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.lab_vars = {}
        
        # Laboratory fields in 2 columns
        fields = [
            ("WBC Count (×10^9/L):", "WBC_Count", 0, 0, "float"),
            ("RBC Count (×10^12/L):", "RBC_Count", 0, 1, "float"),
            ("Hemoglobin (g/dL):", "Hemoglobin", 1, 0, "float"),
            ("RDW (%):", "RDW", 1, 1, "float"),
            ("Segmented Neutro (%):", "Segmented_Neutrophils", 2, 0, "float"),
            ("Thrombocyte (×10^9/L):", "Thrombocyte_Count", 2, 1, "float"),
            ("CRP (mg/L):", "CRP", 3, 0, "float"),
            ("Neutrophilia:", "Neutrophilia", 3, 1, "yesno"),
            ("Ketones in Urine:", "Ketones_in_Urine", 4, 0, "ketones"),
            ("RBC in Urine:", "RBC_in_Urine", 4, 1, "yesno"),
            ("WBC in Urine:", "WBC_in_Urine", 5, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=12, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(segment_frame, textvariable=var, width=16, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=12, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=6)
            elif field_type == "ketones":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["no", "+", "++", "+++"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=12, pady=6)
            
            self.lab_vars[field_name] = var
    
    def collect_input_data(self):
        """Collect all input data from form"""
        
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
                response = messagebox.askyesno("Age Warning", 
                    "This system is designed for pediatric patients (≤18 years). Continue anyway?")
                if not response:
                    return False
            
            return True
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Input validation failed: {e}")
            return False
    
    def predict_diagnosis(self):
        """Make prediction using selected model"""
        
        try:
            # Update status
            self.status_var.set("🔄 Processing precision-focused prediction...")
            self.predict_button.config(state="disabled")
            
            # Collect input data
            input_data = self.collect_input_data()
            
            # Validate input
            if not self.validate_input(input_data):
                self.status_var.set("❌ Validation failed")
                self.predict_button.config(state="normal")
                return
            
            # Make prediction in separate thread
            threading.Thread(target=self._make_prediction, args=(input_data,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_var.set("❌ Error")
            self.predict_button.config(state="normal")
    
    def _make_prediction(self, input_data):
        """Make prediction using real AI models with precision/specificity focus"""
        
        try:
            # Simulate processing time for better UX
            import time
            time.sleep(2.0)
            
            # Get selected model
            model_name = self.model_var.get()
            
            # Use real AI predictor
            prediction, prediction_proba = self.predictor.predict(model_name, input_data)
            
            # Format comprehensive results
            diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
            confidence = prediction_proba[prediction]
            
            # Calculate risk score for display
            risk_score = self._calculate_comprehensive_risk_score(input_data)
            
            results = self._format_precision_focused_results(input_data, model_name, diagnosis, confidence, risk_score, prediction_proba)
            
            # Update results display
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            self.root.after(0, self._display_error, error_msg)
    
    def _format_precision_focused_results(self, input_data, model_name, diagnosis, confidence, risk_score, prediction_proba):
        """Format results from real AI models with precision/specificity focus"""
        
        results = f"""
REAL AI MODEL PREDICTION RESULTS
==================================================

MODEL INFORMATION:
• Model Used: {model_name}
• Model Type: {self.predictor.get_model_info(model_name)['type']}
• Trained: {self.predictor.get_model_info(model_name)['trained']}

PREDICTION RESULTS:
• Diagnosis: {diagnosis}
• Confidence: {confidence:.1%}
• Risk Score: {risk_score:.1f}/100
• Probability (No Appendicitis): {prediction_proba[0]:.1%}
• Probability (Appendicitis): {prediction_proba[1]:.1%}

COMPREHENSIVE PATIENT DATA SUMMARY:
==================================================

DEMOGRAPHIC INFORMATION:
• Age: {input_data.get('Age', 'N/A')} years
• Weight: {input_data.get('Weight', 'N/A')} kg
• Height: {input_data.get('Height', 'N/A')} cm
• BMI: {input_data.get('BMI', 'N/A')}
• Sex: {input_data.get('Sex', 'N/A')}
• Neutrophil %: {input_data.get('Neutrophil_Percentage', 'N/A')}%

CLINICAL SYMPTOMS:
• Body Temperature: {input_data.get('Body_Temperature', 'N/A')}°C
• Lower Right Pain: {input_data.get('Lower_Right_Abd_Pain', 'N/A')}
• Migratory Pain: {input_data.get('Migratory_Pain', 'N/A')}
• Loss of Appetite: {input_data.get('Loss_of_Appetite', 'N/A')}
• Nausea: {input_data.get('Nausea', 'N/A')}
• Coughing Pain: {input_data.get('Coughing_Pain', 'N/A')}
• Dysuria: {input_data.get('Dysuria', 'N/A')}
• Stool Changes: {input_data.get('Stool', 'N/A')}
• Peritonitis: {input_data.get('Peritonitis', 'N/A')}
• Severity: {input_data.get('Severity', 'N/A')}
• Contralateral Rebound: {input_data.get('Contralateral_Rebound_Tenderness', 'N/A')}
• Ipsilateral Rebound: {input_data.get('Ipsilateral_Rebound_Tenderness', 'N/A')}
• Psoas Sign: {input_data.get('Psoas_Sign', 'N/A')}

LABORATORY RESULTS:
• WBC Count: {input_data.get('WBC_Count', 'N/A')} ×10^9/L
• RBC Count: {input_data.get('RBC_Count', 'N/A')} ×10^12/L
• Hemoglobin: {input_data.get('Hemoglobin', 'N/A')} g/dL
• RDW: {input_data.get('RDW', 'N/A')}%
• Segmented Neutrophils: {input_data.get('Segmented_Neutrophils', 'N/A')}%
• Thrombocyte Count: {input_data.get('Thrombocyte_Count', 'N/A')} ×10^9/L
• CRP: {input_data.get('CRP', 'N/A')} mg/L
• Neutrophilia: {input_data.get('Neutrophilia', 'N/A')}
• Ketones in Urine: {input_data.get('Ketones_in_Urine', 'N/A')}
• RBC in Urine: {input_data.get('RBC_in_Urine', 'N/A')}
• WBC in Urine: {input_data.get('WBC_in_Urine', 'N/A')}

MODEL-SPECIFIC PRECISION/SENSITIVITY ANALYSIS:
==================================================
"""
        
        # Add model-specific precision/specificity analysis
        if model_name == 'Transformer':
            results += f"""
TRANSFORMER - HIGHEST PRECISION & SPECIFICITY:
• Precision: {model_stats['precision']:.1%} - Lowest false positive rate among all models
• Specificity: {model_stats['specificity']:.1%} - Best at correctly identifying non-appendicitis cases
• Clinical Impact: Minimizes unnecessary surgeries while maintaining reasonable detection rate
• Best Use Case: Emergency departments where avoiding unnecessary surgeries is critical
"""
        elif model_name == 'Decision Tree':
            results += f"""
DECISION TREE - HIGH SPECIFICITY & GOOD PRECISION:
• Precision: {model_stats['precision']:.1%} - Good positive predictive value
• Specificity: {model_stats['specificity']:.1%} - Second-best at identifying non-appendicitis cases
• Clinical Impact: Clear decision rules, interpretable for clinical staff
• Best Use Case: Situations requiring transparent decision-making process
"""
        elif model_name == 'Gradient Boosting':
            results += f"""
GRADIENT BOOSTING - BALANCED PRECISION/SPECIFICITY:
• Precision: {model_stats['precision']:.1%} - Good balance between precision and recall
• Specificity: {model_stats['specificity']:.1%} - Moderate false positive rate
• Clinical Impact: Robust ensemble method with stable performance across different case types
• Best Use Case: General screening with balanced performance requirements
"""
        elif model_name == 'XGBoost':
            results += f"""
XGBOOST - BALANCED HIGH PRECISION:
• Precision: {model_stats['precision']:.1%} - High precision with good overall accuracy
• Specificity: {model_stats['specificity']:.1%} - Moderate specificity with better sensitivity
• Clinical Impact: Optimized gradient boosting with strong generalization
• Best Use Case: Situations requiring both high precision and reasonable detection rate
"""
        
        results += f"""

CLINICAL INTERPRETATION & RECOMMENDATIONS:
==================================================
"""
        
        if diagnosis == "Appendicitis":
            results += f"""HIGH RISK OF APPENDICITIS DETECTED
CONFIDENCE LEVEL: {confidence:.1%} - PRECISION IMPACT: {confidence * model_stats['precision']:.1%} probability of correct positive prediction

IMMEDIATE CLINICAL RECOMMENDATIONS:
• Emergency surgical consultation urgently needed
• Prepare for possible appendectomy procedure
• Admit for observation and monitoring
• NPO status (nothing by mouth)
• IV fluids and antibiotic therapy
• Pain management as clinically indicated

PRECISION-SPECIFICITY CLINICAL IMPACT:
• False Positive Risk: {(1-confidence)*100:.1f}% - Risk that this is not actually appendicitis
• Based on {model_name} precision: {(1-model_stats['precision'])*100:.1f}% of positive predictions are false positives
• Clinical Decision: Consider confirmatory tests if uncertainty exists

KEY RISK FACTORS IDENTIFIED:
"""
            # List key risk factors with precision context
            if input_data.get('Body_Temperature', 0) > 37.5:
                results += f"     • Elevated temperature ({input_data.get('Body_Temperature')}°C) - Strong positive predictor\n"
            if input_data.get('WBC_Count', 0) > 12:
                results += f"     • Elevated WBC count ({input_data.get('WBC_Count')} ×10^9/L) - High specificity indicator\n"
            if input_data.get('CRP', 0) > 10:
                results += f"     • Elevated CRP ({input_data.get('CRP')} mg/L) - Precise inflammatory marker\n"
            if input_data.get('Peritonitis') == 'yes':
                results += "     • Peritonitis present - Very high specificity finding\n"
                
        else:
            results += f"""LOW RISK OF APPENDICITIS
CONFIDENCE LEVEL: {confidence:.1%} - SPECIFICITY IMPACT: {confidence * model_stats['specificity']:.1%} probability of correct negative prediction

CLINICAL RECOMMENDATIONS:
• Continue observation and monitoring
• Consider alternative diagnoses (gastroenteritis, UTI, etc.)
• Outpatient management may be appropriate
• Re-evaluate if symptoms worsen or change
• Follow-up examination in 24-48 hours
• Symptomatic treatment as needed

PRECISION-SPECIFICITY CLINICAL IMPACT:
• False Negative Risk: {(1-confidence)*100:.1f}% - Risk that appendicitis is actually present
• Based on {model_name} specificity: {(1-model_stats['specificity'])*100:.1f}% of negative predictions are false negatives
• Clinical Decision: Monitor closely for any symptom progression

LOW RISK INDICATORS:
"""
            # List protective factors with specificity context
            if input_data.get('Body_Temperature', 40) <= 37.5:
                results += f"     • Normal temperature ({input_data.get('Body_Temperature')}°C) - High specificity negative indicator\n"
            if input_data.get('WBC_Count', 20) <= 10:
                results += f"     • Normal WBC count ({input_data.get('WBC_Count')} ×10^9/L) - Strong negative predictor\n"
            if input_data.get('CRP', 50) <= 10:
                results += f"     • Normal CRP ({input_data.get('CRP')} mg/L) - High specificity finding\n"
            if input_data.get('Peritonitis') == 'no':
                results += "     • No peritonitis - Very high specificity indicator\n"
        
        results += f"""

MODEL PERFORMANCE SUMMARY:
==================================================
• Precision: {model_stats['precision']:.1%} - {model_stats['precision']*100:.1f}% of positive predictions are correct appendicitis cases
• Specificity: {model_stats['specificity']:.1%} - {model_stats['specificity']*100:.1f}% of negative predictions are correct non-appendicitis cases
• False Positive Rate: {model_stats['fp_rate']:.1%} - {model_stats['fp_rate']*100:.1f}% of healthy patients incorrectly flagged as appendicitis
• False Negative Rate: {model_stats['fn_rate']:.1%} - {model_stats['fn_rate']*100:.1f}% of appendicitis cases missed
• Overall Accuracy: {model_stats['accuracy']:.1%} - {model_stats['accuracy']*100:.1f}% of all predictions are correct
• F1-Score: {model_stats['f1_score']:.1%} - Balance between precision and recall

STUDY FOCUS RECOMMENDATIONS:
==================================================
• For High Precision Requirements: Use Transformer model (75.6% precision) - Minimizes false positives, reduces unnecessary surgeries
• For High Specificity Requirements: Use Transformer model (79.5% specificity) - Best at correctly identifying non-appendicitis cases
• For Balanced Performance: Use XGBoost (75.1% precision, 71.7% specificity) - Good overall clinical utility
• For Interpretability: Use Decision Tree (73.9% precision, 76.4% specificity) - Transparent decision process

MEDICAL DISCLAIMER:
This precision-focused prediction system is for medical reference and decision support only. Clinical judgment should always prevail.
Consider both precision and specificity metrics in clinical decision-making. False positives (unnecessary surgeries) and 
false negatives (missed appendicitis) both have significant clinical consequences.

============================================================
"""
        
        return results
    
    def _calculate_comprehensive_risk_score(self, input_data):
        """Calculate comprehensive risk score"""
        
        risk_score = 0
        
        # Demographic factors
        age = input_data.get('Age', 0)
        if 10 <= age <= 15:
            risk_score += 5
        elif 5 <= age < 10:
            risk_score += 3
        
        # Clinical factors (weighted more heavily)
        temp = input_data.get('Body_Temperature', 36.5)
        if temp > 38.0:
            risk_score += 15
        elif temp > 37.5:
            risk_score += 10
        
        # Pain symptoms
        if input_data.get('Lower_Right_Abd_Pain') == 'yes':
            risk_score += 12
        if input_data.get('Migratory_Pain') == 'yes':
            risk_score += 8
        if input_data.get('Peritonitis') == 'yes':
            risk_score += 20
        if input_data.get('Severity') == 'complicated':
            risk_score += 15
        
        # Other symptoms
        symptom_score = 0
        if input_data.get('Loss_of_Appetite') == 'yes': symptom_score += 3
        if input_data.get('Nausea') == 'yes': symptom_score += 2
        if input_data.get('Ipsilateral_Rebound_Tenderness') == 'yes': symptom_score += 8
        if input_data.get('Psoas_Sign') == 'yes': symptom_score += 5
        risk_score += min(symptom_score, 15)
        
        # Laboratory factors (very important)
        wbc = input_data.get('WBC_Count', 0)
        if wbc > 15:
            risk_score += 15
        elif wbc > 12:
            risk_score += 10
        elif wbc > 10:
            risk_score += 5
        
        crp = input_data.get('CRP', 0)
        if crp > 30:
            risk_score += 15
        elif crp > 15:
            risk_score += 10
        elif crp > 10:
            risk_score += 5
        
        if input_data.get('Neutrophilia') == 'yes':
            risk_score += 8
        
        # Other lab values
        neutro_seg = input_data.get('Segmented_Neutrophils', 0)
        if neutro_seg > 75:
            risk_score += 5
        elif neutro_seg > 70:
            risk_score += 3
        
        # Cap the score at 100
        return min(risk_score, 100)
    
    def _apply_precision_focused_characteristics(self, model_name, risk_score, model_stats):
        """Apply precision/specificity-focused model characteristics"""
        
        # Base probability from risk score
        base_probability = risk_score / 100
        
        # Apply model-specific adjustments focused on precision/specificity
        if model_name == 'Transformer':
            # Highest precision and specificity - more conservative
            adjusted_probability = base_probability * 0.80
        elif model_name == 'Decision Tree':
            # High specificity, good precision
            adjusted_probability = base_probability * 0.85
        elif model_name == 'Gradient Boosting':
            # Balanced precision and specificity
            adjusted_probability = base_probability * 0.90
        elif model_name == 'XGBoost':
            # Balanced with slight precision focus
            adjusted_probability = base_probability * 0.88
        
        # Make prediction
        prediction = 1 if adjusted_probability > 0.5 else 0
        
        # Calculate confidence based on precision/specificity focus
        if prediction == 1:
            # For positive predictions, use precision
            confidence = adjusted_probability * model_stats['precision']
        else:
            # For negative predictions, use specificity
            confidence = (1 - adjusted_probability) * model_stats['specificity']
        
        return prediction, min(confidence, 0.99)
    
    def _display_results(self, results):
        """Display prediction results"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.status_var.set("✅ Precision-focused prediction complete - See detailed analysis on right")
        self.predict_button.config(state="normal")
    
    def _display_error(self, error_msg):
        """Display error message"""
        
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("❌ Error occurred")
        self.predict_button.config(state="normal")
    
    def fill_precision_test(self):
        """Fill form with high precision test case (clear appendicitis)"""
        # This case should test precision - clear negative case
        self.demo_vars['Age'].set(8)
        self.demo_vars['Weight'].set(28)
        self.demo_vars['Height'].set(128)
        self.demo_vars['BMI'].set(17.1)
        self.demo_vars['Sex'].set('Female')
        self.demo_vars['Neutrophil_Percentage'].set(55)
        
        self.clinical_vars['Body_Temperature'].set(37.0)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('no')
        self.clinical_vars['Migratory_Pain'].set('no')
        self.clinical_vars['Loss_of_Appetite'].set('no')
        self.clinical_vars['Nausea'].set('no')
        self.clinical_vars['Peritonitis'].set('no')
        self.clinical_vars['Severity'].set('uncomplicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('no')
        self.clinical_vars['Psoas_Sign'].set('no')
        
        self.lab_vars['WBC_Count'].set(7.8)
        self.lab_vars['CRP'].set(3)
        self.lab_vars['Neutrophilia'].set('no')
        self.lab_vars['Segmented_Neutrophils'].set(52)
        
        self.status_var.set("📝 High precision test case loaded (clear negative)")
    
    def fill_specificity_test(self):
        """Fill form with high specificity test case (clear appendicitis)"""
        # This case should test specificity - clear positive case
        self.demo_vars['Age'].set(12)
        self.demo_vars['Weight'].set(45)
        self.demo_vars['Height'].set(152)
        self.demo_vars['BMI'].set(19.5)
        self.demo_vars['Sex'].set('Male')
        self.demo_vars['Neutrophil_Percentage'].set(75)
        
        self.clinical_vars['Body_Temperature'].set(38.5)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('yes')
        self.clinical_vars['Migratory_Pain'].set('yes')
        self.clinical_vars['Loss_of_Appetite'].set('yes')
        self.clinical_vars['Nausea'].set('yes')
        self.clinical_vars['Peritonitis'].set('yes')
        self.clinical_vars['Severity'].set('complicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('yes')
        self.clinical_vars['Psoas_Sign'].set('yes')
        
        self.lab_vars['WBC_Count'].set(16.5)
        self.lab_vars['CRP'].set(48)
        self.lab_vars['Neutrophilia'].set('yes')
        self.lab_vars['Segmented_Neutrophils'].set(78)
        
        self.status_var.set("📝 High specificity test case loaded (clear positive)")
    
    def fill_balanced_test(self):
        """Fill form with balanced test case (borderline)"""
        # This case should test both precision and specificity
        self.demo_vars['Age'].set(14)
        self.demo_vars['Weight'].set(52)
        self.demo_vars['Height'].set(165)
        self.demo_vars['BMI'].set(19.1)
        self.demo_vars['Sex'].set('Male')
        self.demo_vars['Neutrophil_Percentage'].set(65)
        
        self.clinical_vars['Body_Temperature'].set(37.6)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('yes')
        self.clinical_vars['Migratory_Pain'].set('no')
        self.clinical_vars['Loss_of_Appetite'].set('yes')
        self.clinical_vars['Nausea'].set('yes')
        self.clinical_vars['Peritonitis'].set('no')
        self.clinical_vars['Severity'].set('uncomplicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('no')
        self.clinical_vars['Psoas_Sign'].set('no')
        
        self.lab_vars['WBC_Count'].set(11.2)
        self.lab_vars['CRP'].set(12)
        self.lab_vars['Neutrophilia'].set('no')
        self.lab_vars['Segmented_Neutrophils'].set(65)
        
        self.status_var.set("📝 Balanced precision/specificity test case loaded")
    
    def clear_form(self):
        """Clear all form fields"""
        
        # Reset demographic fields
        for field_name, var in self.demo_vars.items():
            if field_name == 'Sex':
                var.set('Male')
            else:
                var.set(0.0)
        
        # Reset clinical fields
        for field_name, var in self.clinical_vars.items():
            if field_name == 'Body_Temperature':
                var.set(36.5)
            elif field_name == 'Severity':
                var.set('uncomplicated')
            elif field_name == 'Stool':
                var.set('normal')
            else:
                var.set('no')
        
        # Reset laboratory fields
        for field_name, var in self.lab_vars.items():
            if isinstance(var, tk.StringVar):
                if field_name == 'Ketones_in_Urine':
                    var.set('no')
                else:
                    var.set('no')
            else:
                var.set(0.0)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("🧹 Form cleared - All fields reset to defaults")

def main():
    """Main function to run precision-focused GUI application"""
    
    root = tk.Tk()
    app = PrecisionFocusedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
