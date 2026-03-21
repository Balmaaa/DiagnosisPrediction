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

class SingleFormAppendicitisGUI:
    """Single-form GUI with real AI models and comprehensive medical data input"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System - Real AI Models")
        self.root.geometry("1600x1000")
        
        # Initialize real AI predictor
        self.predictor = AppendicitisPredictor()
        
        # Create main layout
        self.create_widgets()
    
    def create_widgets(self):
        """Create single-form GUI widgets with 3 segments"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System", 
                                font=('Arial', 18, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Complete Medical Assessment - Single Form Interface", 
                                  font=('Arial', 12, 'italic'))
        subtitle_label.pack()
        
        # Model Selection with Real AI Models
        model_frame = ttk.LabelFrame(self.root, text="Real AI Model Selection", padding="10")
        model_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="Transformer")
        
        # Get available models from real predictor
        available_models = self.predictor.get_available_models()
        
        for i, model in enumerate(available_models):
            model_info = self.predictor.get_model_info(model)
            model_text = f"{model}\n({model_info['type']} - Trained: {model_info['trained']})"
            ttk.Radiobutton(model_frame, text=model_text, variable=self.model_var, 
                           value=model).pack(side=tk.LEFT, padx=20)
        
        # Create main scrollable form
        self.create_single_form()
        
        # Prediction button and results
        self.create_prediction_section()
        
    def create_single_form(self):
        """Create single form with 3 segments"""
        
        # Main form container with scrollbar
        main_form_frame = ttk.LabelFrame(self.root, text="Patient Medical Information", padding="10")
        main_form_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(main_form_frame)
        scrollbar = ttk.Scrollbar(main_form_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # SEGMENT 1: DEMOGRAPHIC INFORMATION
        self.create_demographic_segment(scrollable_frame)
        
        # SEGMENT 2: CLINICAL SYMPTOMS  
        self.create_clinical_segment(scrollable_frame)
        
        # SEGMENT 3: LABORATORY RESULTS
        self.create_laboratory_segment(scrollable_frame)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_demographic_segment(self, parent):
        """Create demographic information segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="📋 SEGMENT 1: DEMOGRAPHIC INFORMATION", padding="15")
        segment_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.demo_vars = {}
        
        # Demographic fields in 3 columns
        fields = [
            ("Age (years):", "Age", 0, 0, "float"),
            ("Weight (kg):", "Weight", 0, 1, "float"),
            ("Height (cm):", "Height", 0, 2, "float"),
            ("BMI:", "BMI", 1, 0, "float"),
            ("Sex:", "Sex", 1, 1, "sex"),
            ("Neutrophil %:", "Neutrophil_Percentage", 1, 2, "float")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=15, pady=8, sticky=tk.W)
            
            if field_type == "sex":
                var = tk.StringVar(value="Male")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["Male", "Female"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=8)
            else:
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(segment_frame, textvariable=var, width=18, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=15, pady=8)
            
            self.demo_vars[field_name] = var
    
    def create_clinical_segment(self, parent):
        """Create clinical symptoms segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="🏥 SEGMENT 2: CLINICAL SYMPTOMS", padding="15")
        segment_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.clinical_vars = {}
        
        # Clinical symptoms in 3 columns
        fields = [
            ("Body Temp (°C):", "Body_Temperature", 0, 0, "float"),
            ("Lower Right Pain:", "Lower_Right_Abd_Pain", 0, 1, "yesno"),
            ("Migratory Pain:", "Migratory_Pain", 0, 2, "yesno"),
            ("Loss of Appetite:", "Loss_of_Appetite", 1, 0, "yesno"),
            ("Nausea:", "Nausea", 1, 1, "yesno"),
            ("Coughing Pain:", "Coughing_Pain", 1, 2, "yesno"),
            ("Dysuria:", "Dysuria", 2, 0, "yesno"),
            ("Stool Changes:", "Stool", 2, 1, "stool"),
            ("Peritonitis:", "Peritonitis", 2, 2, "yesno"),
            ("Severity:", "Severity", 3, 0, "severity"),
            ("Contralateral Rebound:", "Contralateral_Rebound_Tenderness", 3, 1, "yesno"),
            ("Ipsilateral Rebound:", "Ipsilateral_Rebound_Tenderness", 3, 2, "yesno"),
            ("Psoas Sign:", "Psoas_Sign", 4, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=15, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=36.5)
                entry = ttk.Entry(segment_frame, textvariable=var, width=18, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=15, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=6)
            elif field_type == "severity":
                var = tk.StringVar(value="uncomplicated")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["uncomplicated", "complicated"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=6)
            elif field_type == "stool":
                var = tk.StringVar(value="normal")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["normal", "constipation", "diarrhea"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=6)
            
            self.clinical_vars[field_name] = var
    
    def create_laboratory_segment(self, parent):
        """Create laboratory results segment"""
        
        segment_frame = ttk.LabelFrame(parent, text="🔬 SEGMENT 3: LABORATORY RESULTS", padding="15")
        segment_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=10)
        
        self.lab_vars = {}
        
        # Laboratory fields in 3 columns
        fields = [
            ("WBC Count (×10^9/L):", "WBC_Count", 0, 0, "float"),
            ("RBC Count (×10^12/L):", "RBC_Count", 0, 1, "float"),
            ("Hemoglobin (g/dL):", "Hemoglobin", 0, 2, "float"),
            ("RDW (%):", "RDW", 1, 0, "float"),
            ("Segmented Neutro (%):", "Segmented_Neutrophils", 1, 1, "float"),
            ("Thrombocyte (×10^9/L):", "Thrombocyte_Count", 1, 2, "float"),
            ("CRP (mg/L):", "CRP", 2, 0, "float"),
            ("Neutrophilia:", "Neutrophilia", 2, 1, "yesno"),
            ("Ketones in Urine:", "Ketones_in_Urine", 2, 2, "ketones"),
            ("RBC in Urine:", "RBC_in_Urine", 3, 0, "yesno"),
            ("WBC in Urine:", "WBC_in_Urine", 3, 1, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(segment_frame, text=label_text, font=('Arial', 11, 'bold'))
            label.grid(row=row, column=col*2, padx=15, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(segment_frame, textvariable=var, width=18, font=('Arial', 10))
                entry.grid(row=row, column=col*2+1, padx=15, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=6)
            elif field_type == "ketones":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(segment_frame, textvariable=var, 
                                   values=["no", "+", "++", "+++"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=15, pady=6)
            
            self.lab_vars[field_name] = var
    
    def create_prediction_section(self):
        """Create prediction button and results section"""
        
        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=3, pady=15)
        
        # Style buttons
        style = ttk.Style()
        style.configure("Predict.TButton", font=('Arial', 12, 'bold'), padding=10)
        style.configure("Clear.TButton", font=('Arial', 10), padding=8)
        
        self.predict_button = ttk.Button(button_frame, text="🔍 Predict Diagnosis", 
                                        command=self.predict_diagnosis, style="Predict.TButton")
        self.predict_button.pack(side=tk.LEFT, padx=20)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="🧹 Clear Form", command=self.clear_form, style="Clear.TButton")
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Quick fill buttons for testing
        test_frame = ttk.Frame(button_frame)
        test_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(test_frame, text="Quick Test:", font=('Arial', 10, 'italic')).pack(side=tk.LEFT, padx=5)
        ttk.Button(test_frame, text="High Risk", command=self.fill_high_risk).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_frame, text="Low Risk", command=self.fill_low_risk).pack(side=tk.LEFT, padx=2)
        ttk.Button(test_frame, text="Borderline", command=self.fill_borderline).pack(side=tk.LEFT, padx=2)
        
        # Results section
        results_frame = ttk.LabelFrame(self.root, text="📊 PREDICTION RESULTS", padding="15")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=120, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Single form with all medical indicators loaded")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, font=('Arial', 10))
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
    
    def collect_input_data(self):
        """Collect all input data from the single form"""
        
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
            self.status_var.set("🔄 Processing prediction...")
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
        """Make prediction using real AI models"""
        
        try:
            # Simulate processing time for better UX
            import time
            time.sleep(1.5)
            
            # Get selected model
            model_name = self.model_var.get()
            
            # Use real AI predictor
            prediction, prediction_proba = self.predictor.predict(model_name, input_data)
            
            # Format comprehensive results
            diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
            confidence = prediction_proba[prediction]
            
            # Calculate risk score for display
            risk_score = self._calculate_comprehensive_risk_score(input_data)
            
            results = f"""
REAL AI MODEL PREDICTION RESULTS
==================================================

MODEL INFORMATION:
• Model Used: {model_name}
• Model Type: {self.predictor.get_model_info(model_name)['type']}
• Trained: {self.predictor.get_model_info(model_name)['trained']}

PREDICTION RESULTS:
• Diagnosis: {diagnosis}
• Confidence: {confidence:.2%}
• Risk Score: {risk_score}/100
• Probability (No Appendicitis): {prediction_proba[0]:.2%}
• Probability (Appendicitis): {prediction_proba[1]:.2%}

{'='*100}
COMPREHENSIVE PATIENT DATA SUMMARY
{'='*100}

📋 DEMOGRAPHIC INFORMATION:
   • Age: {input_data.get('Age', 'N/A')} years
   • Weight: {input_data.get('Weight', 'N/A')} kg
   • Height: {input_data.get('Height', 'N/A')} cm
   • BMI: {input_data.get('BMI', 'N/A')}
   • Sex: {input_data.get('Sex', 'N/A')}
   • Neutrophil %: {input_data.get('Neutrophil_Percentage', 'N/A')}%

🏥 CLINICAL SYMPTOMS:
   • Body Temperature: {input_data.get('Body_Temperature', 'N/A')}°C
   • Lower Right Abdominal Pain: {input_data.get('Lower_Right_Abd_Pain', 'N/A')}
   • Migratory Pain: {input_data.get('Migratory_Pain', 'N/A')}
   • Loss of Appetite: {input_data.get('Loss_of_Appetite', 'N/A')}
   • Nausea: {input_data.get('Nausea', 'N/A')}
   • Coughing Pain: {input_data.get('Coughing_Pain', 'N/A')}
   • Dysuria: {input_data.get('Dysuria', 'N/A')}
   • Stool Changes: {input_data.get('Stool', 'N/A')}
   • Peritonitis: {input_data.get('Peritonitis', 'N/A')}
   • Severity: {input_data.get('Severity', 'N/A')}
   • Contralateral Rebound Tenderness: {input_data.get('Contralateral_Rebound_Tenderness', 'N/A')}
   • Ipsilateral Rebound Tenderness: {input_data.get('Ipsilateral_Rebound_Tenderness', 'N/A')}
   • Psoas Sign: {input_data.get('Psoas_Sign', 'N/A')}

🔬 LABORATORY RESULTS:
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

{'='*100}
MODEL-SPECIFIC ANALYSIS
{'='*100}
"""
            
            # Add model-specific analysis
            if model_name == 'XGBoost':
                results += """
🔹 XGBoost Analysis:
   • Best overall accuracy (73.08%)
   • Balanced sensitivity/specificity
   • Optimal for general screening
   • Reliable for complex cases
"""
            elif model_name == 'Gradient Boosting':
                results += """
🔹 Gradient Boosting Analysis:
   • Strong ensemble method (72.44% accuracy)
   • Slightly conservative predictions
   • Good generalization capability
   • Robust for varied presentations
"""
            elif model_name == 'Decision Tree':
                results += """
🔹 Decision Tree Analysis:
   • Interpretable decision rules (70.51% accuracy)
   • Clear clinical decision pathways
   • Good for understanding key factors
   • Rule-based medical reasoning
"""
            elif model_name == 'Transformer':
                results += """
🔹 Transformer Analysis:
   • Deep learning approach (69.87% accuracy)
   • Highest specificity (79.53%)
   • Best for ruling out appendicitis
   • Conservative prediction style
"""
            
            results += f"""
{'='*100}
MEDICAL INTERPRETATION & RECOMMENDATIONS
{'='*100}
"""
            
            if prediction == 1:
                results += f"""
⚠️  HIGH RISK OF APPENDICITIS DETECTED
🎯 Confidence Level: {confidence:.1%}

🚨 IMMEDIATE CLINICAL RECOMMENDATIONS:
   • Emergency medical evaluation required
   • Surgical consultation urgently needed
   • Prepare for possible appendectomy
   • Admit for observation and monitoring
   • NPO status (nothing by mouth)
   • IV fluids for hydration
   • Pain management as appropriate
   • Antibiotic therapy if indicated

🔍 KEY RISK FACTORS IDENTIFIED:
"""
                # List key risk factors
                if input_data.get('Body_Temperature', 0) > 37.5:
                    results += f"   • Elevated temperature ({input_data.get('Body_Temperature')}°C)\n"
                if input_data.get('WBC_Count', 0) > 12:
                    results += f"   • Elevated WBC count ({input_data.get('WBC_Count')} ×10^9/L)\n"
                if input_data.get('CRP', 0) > 10:
                    results += f"   • Elevated CRP ({input_data.get('CRP')} mg/L)\n"
                if input_data.get('Peritonitis') == 'yes':
                    results += "   • Peritonitis present\n"
                if input_data.get('Severity') == 'complicated':
                    results += "   • Complicated presentation\n"
                
            else:
                results += f"""
✅ LOW RISK OF APPENDICITIS
🎯 Confidence Level: {confidence:.1%}

🏥 CLINICAL RECOMMENDATIONS:
   • Continue observation and monitoring
   • Consider alternative diagnoses
   • Outpatient management may be appropriate
   • Re-evaluate if symptoms worsen
   • Follow-up examination in 24-48 hours
   • Symptomatic treatment as needed
   • Patient education on warning signs

🛡️ LOW RISK INDICATORS:
"""
                # List protective factors
                if input_data.get('Body_Temperature', 40) <= 37.5:
                    results += f"   • Normal temperature ({input_data.get('Body_Temperature')}°C)\n"
                if input_data.get('WBC_Count', 20) <= 10:
                    results += f"   • Normal WBC count ({input_data.get('WBC_Count')} ×10^9/L)\n"
                if input_data.get('CRP', 50) <= 10:
                    results += f"   • Normal CRP ({input_data.get('CRP')} mg/L)\n"
                if input_data.get('Peritonitis') == 'no':
                    results += "   • No peritonitis\n"
                if input_data.get('Severity') == 'uncomplicated':
                    results += "   • Uncomplicated presentation\n"
            
            results += f"""
{'='*100}
⚕️  MEDICAL DISCLAIMER
{'='*100}
This prediction system is for medical reference and decision support only.
Clinical judgment should always prevail in patient care decisions.
This tool complements, but does not replace, medical expertise.
Always consider complete clinical context and patient history.

{'='*100}
📊 MODEL PERFORMANCE SUMMARY
{'='*100}
• Overall Accuracy: {model_stats['accuracy']:.1%}
• True Positive Rate (Sensitivity): {model_stats['sensitivity']:.1%}
• True Negative Rate (Specificity): {model_stats['specificity']:.1%}
• Model Type: {model_name}
• Risk Assessment: {diagnosis}
• Confidence Level: {confidence:.1%}

{'='*100}
"""
            
            # Update results display
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            self.root.after(0, self._display_error, error_msg)
    
    def _calculate_comprehensive_risk_score(self, input_data):
        """Calculate comprehensive risk score using all available features"""
        
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
    
    def _apply_model_characteristics(self, model_name, risk_score, model_stats):
        """Apply model-specific characteristics to prediction"""
        
        # Base probability from risk score
        base_probability = risk_score / 100
        
        # Apply model-specific adjustments
        if model_name == 'XGBoost':
            # Most balanced
            adjusted_probability = base_probability
        elif model_name == 'Gradient Boosting':
            # Slightly more conservative
            adjusted_probability = base_probability * 0.95
        elif model_name == 'Decision Tree':
            # More variable based on clear rules
            adjusted_probability = base_probability * 1.05
        elif model_name == 'Transformer':
            # Higher specificity (more likely to say no)
            adjusted_probability = base_probability * 0.85
        
        # Make prediction
        prediction = 1 if adjusted_probability > 0.5 else 0
        
        # Calculate confidence based on model characteristics
        if prediction == 1:
            confidence = adjusted_probability * model_stats['sensitivity']
        else:
            confidence = (1 - adjusted_probability) * model_stats['specificity']
        
        return prediction, min(confidence, 0.99)
    
    def _display_results(self, results):
        """Display prediction results"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.status_var.set("✅ Prediction complete - See detailed results below")
        self.predict_button.config(state="normal")
    
    def _display_error(self, error_msg):
        """Display error message"""
        
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("❌ Error occurred")
        self.predict_button.config(state="normal")
    
    def fill_high_risk(self):
        """Fill form with high-risk appendicitis case"""
        # Demographic
        self.demo_vars['Age'].set(12)
        self.demo_vars['Weight'].set(45)
        self.demo_vars['Height'].set(152)
        self.demo_vars['BMI'].set(19.5)
        self.demo_vars['Sex'].set('Male')
        self.demo_vars['Neutrophil_Percentage'].set(75)
        
        # Clinical
        self.clinical_vars['Body_Temperature'].set(38.5)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('yes')
        self.clinical_vars['Migratory_Pain'].set('yes')
        self.clinical_vars['Loss_of_Appetite'].set('yes')
        self.clinical_vars['Nausea'].set('yes')
        self.clinical_vars['Peritonitis'].set('yes')
        self.clinical_vars['Severity'].set('complicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('yes')
        self.clinical_vars['Psoas_Sign'].set('yes')
        
        # Laboratory
        self.lab_vars['WBC_Count'].set(16.5)
        self.lab_vars['CRP'].set(48)
        self.lab_vars['Neutrophilia'].set('yes')
        self.lab_vars['Segmented_Neutrophils'].set(78)
        
        self.status_var.set("📝 High-risk test case loaded")
    
    def fill_low_risk(self):
        """Fill form with low-risk case"""
        # Demographic
        self.demo_vars['Age'].set(8)
        self.demo_vars['Weight'].set(28)
        self.demo_vars['Height'].set(128)
        self.demo_vars['BMI'].set(17.1)
        self.demo_vars['Sex'].set('Female')
        self.demo_vars['Neutrophil_Percentage'].set(55)
        
        # Clinical
        self.clinical_vars['Body_Temperature'].set(37.0)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('no')
        self.clinical_vars['Migratory_Pain'].set('no')
        self.clinical_vars['Loss_of_Appetite'].set('no')
        self.clinical_vars['Nausea'].set('no')
        self.clinical_vars['Peritonitis'].set('no')
        self.clinical_vars['Severity'].set('uncomplicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('no')
        self.clinical_vars['Psoas_Sign'].set('no')
        
        # Laboratory
        self.lab_vars['WBC_Count'].set(7.8)
        self.lab_vars['CRP'].set(3)
        self.lab_vars['Neutrophilia'].set('no')
        self.lab_vars['Segmented_Neutrophils'].set(52)
        
        self.status_var.set("📝 Low-risk test case loaded")
    
    def fill_borderline(self):
        """Fill form with borderline case"""
        # Demographic
        self.demo_vars['Age'].set(14)
        self.demo_vars['Weight'].set(52)
        self.demo_vars['Height'].set(165)
        self.demo_vars['BMI'].set(19.1)
        self.demo_vars['Sex'].set('Male')
        self.demo_vars['Neutrophil_Percentage'].set(65)
        
        # Clinical
        self.clinical_vars['Body_Temperature'].set(37.6)
        self.clinical_vars['Lower_Right_Abd_Pain'].set('yes')
        self.clinical_vars['Migratory_Pain'].set('no')
        self.clinical_vars['Loss_of_Appetite'].set('yes')
        self.clinical_vars['Nausea'].set('yes')
        self.clinical_vars['Peritonitis'].set('no')
        self.clinical_vars['Severity'].set('uncomplicated')
        self.clinical_vars['Ipsilateral_Rebound_Tenderness'].set('no')
        self.clinical_vars['Psoas_Sign'].set('no')
        
        # Laboratory
        self.lab_vars['WBC_Count'].set(11.2)
        self.lab_vars['CRP'].set(12)
        self.lab_vars['Neutrophilia'].set('no')
        self.lab_vars['Segmented_Neutrophils'].set(65)
        
        self.status_var.set("📝 Borderline test case loaded")
    
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
    """Main function to run the single-form GUI application"""
    
    root = tk.Tk()
    app = SingleFormAppendicitisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
