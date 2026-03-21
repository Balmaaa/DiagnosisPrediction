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

class ComprehensiveAppendicitisGUI:
    """Complete GUI with real AI models and all dataset features"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System - Real AI Models")
        self.root.geometry("1400x900")
        
        # Initialize real AI predictor
        self.predictor = AppendicitisPredictor()
        
        # Create main layout
        self.create_widgets()
    
    def create_widgets(self):
        """Create comprehensive GUI widgets"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System", 
                                font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Complete Dataset Features - All Medical Indicators", 
                                  font=('Arial', 10, 'italic'))
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
                           value=model).pack(side=tk.LEFT, padx=15)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # Create comprehensive tabs
        self.create_demographic_tab()
        self.create_clinical_tab()
        self.create_laboratory_tab()
        
        # Prediction button and results
        self.create_prediction_section()
    
    def create_demographic_tab(self):
        """Create Demographic Information tab"""
        
        demo_frame = ttk.Frame(self.notebook)
        self.notebook.add(demo_frame, text="Demographic Information")
        
        # Create scrollable frame
        canvas = tk.Canvas(demo_frame)
        scrollbar = ttk.Scrollbar(demo_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.demo_vars = {}
        
        # Demographic fields
        fields = [
            ("Age (years):", "Age", 0, 0, "float"),
            ("Weight (kg):", "Weight", 0, 1, "float"),
            ("Height (cm):", "Height", 1, 0, "float"),
            ("BMI:", "BMI", 1, 1, "float"),
            ("Sex:", "Sex", 2, 0, "sex"),
            ("Neutrophil Percentage (%):", "Neutrophil_Percentage", 2, 1, "float")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text, font=('Arial', 10, 'bold'))
            label.grid(row=row, column=col*2, padx=10, pady=8, sticky=tk.W)
            
            if field_type == "sex":
                var = tk.StringVar(value="Male")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["Male", "Female"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=8)
            else:
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=18)
                entry.grid(row=row, column=col*2+1, padx=10, pady=8)
            
            self.demo_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_clinical_tab(self):
        """Create Clinical Symptoms tab with ALL clinical features"""
        
        clinical_frame = ttk.Frame(self.notebook)
        self.notebook.add(clinical_frame, text="Clinical Symptoms")
        
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
        
        self.clinical_vars = {}
        
        # ALL clinical symptoms from dataset
        fields = [
            ("Body Temperature (°C):", "Body_Temperature", 0, 0, "float"),
            ("Lower Right Abdominal Pain:", "Lower_Right_Abd_Pain", 0, 1, "yesno"),
            ("Migratory Pain:", "Migratory_Pain", 1, 0, "yesno"),
            ("Loss of Appetite:", "Loss_of_Appetite", 1, 1, "yesno"),
            ("Nausea:", "Nausea", 2, 0, "yesno"),
            ("Coughing Pain:", "Coughing_Pain", 2, 1, "yesno"),
            ("Dysuria:", "Dysuria", 3, 0, "yesno"),
            ("Stool Changes:", "Stool", 3, 1, "stool"),
            ("Peritonitis:", "Peritonitis", 4, 0, "yesno"),
            ("Severity:", "Severity", 4, 1, "severity"),
            ("Contralateral Rebound Tenderness:", "Contralateral_Rebound_Tenderness", 5, 0, "yesno"),
            ("Ipsilateral Rebound Tenderness:", "Ipsilateral_Rebound_Tenderness", 5, 1, "yesno"),
            ("Psoas Sign:", "Psoas_Sign", 6, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text, font=('Arial', 10, 'bold'))
            label.grid(row=row, column=col*2, padx=10, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=36.5)
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=18)
                entry.grid(row=row, column=col*2+1, padx=10, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=6)
            elif field_type == "severity":
                var = tk.StringVar(value="uncomplicated")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["uncomplicated", "complicated"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=6)
            elif field_type == "stool":
                var = tk.StringVar(value="normal")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["normal", "constipation", "diarrhea"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=6)
            
            self.clinical_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_laboratory_tab(self):
        """Create Laboratory Results tab with ALL lab features"""
        
        lab_frame = ttk.Frame(self.notebook)
        self.notebook.add(lab_frame, text="Laboratory Results")
        
        # Create scrollable frame
        canvas = tk.Canvas(lab_frame)
        scrollbar = ttk.Scrollbar(lab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.lab_vars = {}
        
        # ALL laboratory features from dataset
        fields = [
            ("WBC Count (×10^9/L):", "WBC_Count", 0, 0, "float"),
            ("RBC Count (×10^12/L):", "RBC_Count", 0, 1, "float"),
            ("Hemoglobin (g/dL):", "Hemoglobin", 1, 0, "float"),
            ("RDW (%):", "RDW", 1, 1, "float"),
            ("Segmented Neutrophils (%):", "Segmented_Neutrophils", 2, 0, "float"),
            ("Thrombocyte Count (×10^9/L):", "Thrombocyte_Count", 2, 1, "float"),
            ("CRP (mg/L):", "CRP", 3, 0, "float"),
            ("Neutrophilia:", "Neutrophilia", 3, 1, "yesno"),
            ("Ketones in Urine:", "Ketones_in_Urine", 4, 0, "ketones"),
            ("RBC in Urine:", "RBC_in_Urine", 4, 1, "yesno"),
            ("WBC in Urine:", "WBC_in_Urine", 5, 0, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text, font=('Arial', 10, 'bold'))
            label.grid(row=row, column=col*2, padx=10, pady=6, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=18)
                entry.grid(row=row, column=col*2+1, padx=10, pady=6)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["yes", "no"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=6)
            elif field_type == "ketones":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, 
                                   values=["no", "+", "++", "+++"], state="readonly", width=15)
                combo.grid(row=row, column=col*2+1, padx=10, pady=6)
            
            self.lab_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_prediction_section(self):
        """Create prediction button and results section"""
        
        # Prediction button
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.predict_button = ttk.Button(button_frame, text="Predict Diagnosis", 
                                        command=self.predict_diagnosis)
        self.predict_button.pack(side=tk.LEFT, padx=10)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear Form", command=self.clear_form)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Results section
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, height=12, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Complete dataset features loaded")
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
                self.status_var.set("Validation failed")
                self.predict_button.config(state="normal")
                return
            
            # Make prediction in separate thread
            threading.Thread(target=self._make_prediction, args=(input_data,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_var.set("Error")
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

{'='*80}
COMPREHENSIVE PATIENT DATA SUMMARY
{'='*80}
CONFIDENCE: {confidence:.2%}
RISK SCORE: {risk_score:.1f}/100

COMPREHENSIVE INPUT SUMMARY:
{'='*80}

DEMOGRAPHIC INFORMATION:
- Age: {input_data.get('Age', 'N/A')} years
- Weight: {input_data.get('Weight', 'N/A')} kg
- Height: {input_data.get('Height', 'N/A')} cm
- BMI: {input_data.get('BMI', 'N/A')}
- Sex: {input_data.get('Sex', 'N/A')}
- Neutrophil %: {input_data.get('Neutrophil_Percentage', 'N/A')}%

CLINICAL SYMPTOMS:
- Body Temperature: {input_data.get('Body_Temperature', 'N/A')}°C
- Lower Right Abdominal Pain: {input_data.get('Lower_Right_Abd_Pain', 'N/A')}
- Migratory Pain: {input_data.get('Migratory_Pain', 'N/A')}
- Loss of Appetite: {input_data.get('Loss_of_Appetite', 'N/A')}
- Nausea: {input_data.get('Nausea', 'N/A')}
- Coughing Pain: {input_data.get('Coughing_Pain', 'N/A')}
- Dysuria: {input_data.get('Dysuria', 'N/A')}
- Stool Changes: {input_data.get('Stool', 'N/A')}
- Peritonitis: {input_data.get('Peritonitis', 'N/A')}
- Severity: {input_data.get('Severity', 'N/A')}
- Contralateral Rebound Tenderness: {input_data.get('Contralateral_Rebound_Tenderness', 'N/A')}
- Ipsilateral Rebound Tenderness: {input_data.get('Ipsilateral_Rebound_Tenderness', 'N/A')}
- Psoas Sign: {input_data.get('Psoas_Sign', 'N/A')}

LABORATORY RESULTS:
- WBC Count: {input_data.get('WBC_Count', 'N/A')} ×10^9/L
- RBC Count: {input_data.get('RBC_Count', 'N/A')} ×10^12/L
- Hemoglobin: {input_data.get('Hemoglobin', 'N/A')} g/dL
- RDW: {input_data.get('RDW', 'N/A')}%
- Segmented Neutrophils: {input_data.get('Segmented_Neutrophils', 'N/A')}%
- Thrombocyte Count: {input_data.get('Thrombocyte_Count', 'N/A')} ×10^9/L
- CRP: {input_data.get('CRP', 'N/A')} mg/L
- Neutrophilia: {input_data.get('Neutrophilia', 'N/A')}
- Ketones in Urine: {input_data.get('Ketones_in_Urine', 'N/A')}
- RBC in Urine: {input_data.get('RBC_in_Urine', 'N/A')}
- WBC in Urine: {input_data.get('WBC_in_Urine', 'N/A')}

PREDICTION ANALYSIS:
{'='*80}
"""
            
            # Add model-specific analysis
            if model_name == 'XGBoost':
                results += "XGBoost Analysis: High accuracy model with balanced sensitivity/specificity.\n"
                results += "Best for overall screening and general diagnosis.\n"
            elif model_name == 'Gradient Boosting':
                results += "Gradient Boosting Analysis: Strong ensemble method with good generalization.\n"
                results += "Reliable for complex cases with multiple risk factors.\n"
            elif model_name == 'Decision Tree':
                results += "Decision Tree Analysis: Interpretable model with clear decision rules.\n"
                results += "Good for understanding key decision factors.\n"
            elif model_name == 'Transformer':
                results += "Transformer Analysis: Deep learning model with highest specificity.\n"
                results += "Best for ruling out appendicitis (high specificity).\n"
            
            results += f"""
MEDICAL INTERPRETATION:
{'='*80}
"""
            
            if prediction == 1:
                results += f"""
⚠️  HIGH RISK OF APPENDICITIS DETECTED
Confidence Level: {confidence:.1%}

CLINICAL RECOMMENDATIONS:
- Immediate medical evaluation required
- Emergency surgical consultation recommended
- Prepare for possible appendectomy
- Admit for observation and monitoring
- IV fluids and NPO status
- Pain management as appropriate

KEY RISK FACTORS IDENTIFIED:
"""
                # List key risk factors
                if input_data.get('Body_Temperature', 0) > 37.5:
                    results += f"- Elevated temperature ({input_data.get('Body_Temperature')}°C)\n"
                if input_data.get('WBC_Count', 0) > 12:
                    results += f"- Elevated WBC count ({input_data.get('WBC_Count')} ×10^9/L)\n"
                if input_data.get('CRP', 0) > 10:
                    results += f"- Elevated CRP ({input_data.get('CRP')} mg/L)\n"
                if input_data.get('Peritonitis') == 'yes':
                    results += "- Peritonitis present\n"
                if input_data.get('Lower_Right_Abd_Pain') == 'yes':
                    results += "- Localized RLQ pain\n"
                
            else:
                results += f"""
✅ LOW RISK OF APPENDICITIS
Confidence Level: {confidence:.1%}

CLINICAL RECOMMENDATIONS:
- Continue observation and monitoring
- Consider alternative diagnoses
- Outpatient management may be appropriate
- Re-evaluate if symptoms worsen
- Follow-up examination in 24-48 hours

LOW RISK INDICATORS:
"""
                # List protective factors
                if input_data.get('Body_Temperature', 40) <= 37.5:
                    results += f"- Normal temperature ({input_data.get('Body_Temperature')}°C)\n"
                if input_data.get('WBC_Count', 20) <= 10:
                    results += f"- Normal WBC count ({input_data.get('WBC_Count')} ×10^9/L)\n"
                if input_data.get('CRP', 50) <= 10:
                    results += f"- Normal CRP ({input_data.get('CRP')} mg/L)\n"
                if input_data.get('Peritonitis') == 'no':
                    results += "- No peritonitis\n"
            
            results += f"""
{'='*80}
MODEL PERFORMANCE SUMMARY:
- Overall Accuracy: {model_stats['accuracy']:.1%}
- True Positive Rate (Sensitivity): {model_stats['sensitivity']:.1%}
- True Negative Rate (Specificity): {model_stats['specificity']:.1%}

IMPORTANT MEDICAL DISCLAIMER:
This prediction system is for medical reference and decision support only.
Clinical judgment should always prevail in patient care decisions.
This tool complements, but does not replace, medical expertise.
{'='*80}
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
            # Balanced performance
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
        self.status_var.set("Prediction complete")
        self.predict_button.config(state="normal")
    
    def _display_error(self, error_msg):
        """Display error message"""
        
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("Error")
        self.predict_button.config(state="normal")
    
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
        self.status_var.set("Form cleared - All fields reset")

def main():
    """Main function to run the comprehensive GUI application"""
    
    root = tk.Tk()
    app = ComprehensiveAppendicitisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
