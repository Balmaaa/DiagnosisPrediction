import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from pathlib import Path
import threading

# Import the real backend predictor
from backend_predictor import AppendicitisPredictor

class SimpleAppendicitisGUI:
    """Simplified GUI with real AI models"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Pediatric Appendicitis Prediction System - Real AI Models")
        self.root.geometry("800x600")
        
        # Initialize real AI predictor
        self.predictor = AppendicitisPredictor()
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create main GUI widgets"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(title_frame, text="Pediatric Appendicitis Prediction System", 
                                font=('Arial', 16, 'bold'))
        title_label.pack()
        
        # Model Selection with Real AI Models
        model_frame = ttk.LabelFrame(self.root, text="Real AI Model Selection", padding="10")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="Transformer")
        
        # Get available models from real predictor
        available_models = self.predictor.get_available_models()
        
        for i, model in enumerate(available_models):
            model_info = self.predictor.get_model_info(model)
            model_text = f"{model}\n({model_info['type']})"
            ttk.Radiobutton(model_frame, text=model_text, variable=self.model_var, 
                           value=model).grid(row=0, column=i, padx=10)
        
        # Input Form
        form_frame = ttk.LabelFrame(self.root, text="Patient Information", padding="10")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create scrollable frame
        canvas = tk.Canvas(form_frame)
        scrollbar = ttk.Scrollbar(form_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Input fields
        self.input_vars = {}
        
        # Demographic Information
        ttk.Label(scrollable_frame, text="DEMOGRAPHIC INFORMATION", font=('Arial', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        fields = [
            ("Age (years):", "Age", 1, 0, "float"),
            ("Weight (kg):", "Weight", 1, 1, "float"),
            ("Height (cm):", "Height", 2, 0, "float"),
            ("Sex:", "Sex", 2, 1, "sex"),
            ("Body Temperature (°C):", "Body_Temperature", 3, 0, "float"),
            ("WBC Count (×10^9/L):", "WBC_Count", 3, 1, "float"),
            ("CRP (mg/L):", "CRP", 4, 0, "float"),
            ("Appendix Diameter (mm):", "Appendix_Diameter", 4, 1, "float"),
            ("Lower Right Abdominal Pain:", "Lower_Right_Abd_Pain", 5, 0, "yesno"),
            ("Nausea:", "Nausea", 5, 1, "yesno"),
            ("Peritonitis:", "Peritonitis", 6, 0, "yesno"),
            ("US Performed:", "US_Performed", 6, 1, "yesno")
        ]
        
        for label_text, field_name, row, col, field_type in fields:
            label = ttk.Label(scrollable_frame, text=label_text)
            label.grid(row=row, column=col*2, padx=5, pady=5, sticky=tk.W)
            
            if field_type == "float":
                var = tk.DoubleVar(value=0.0)
                entry = ttk.Entry(scrollable_frame, textvariable=var, width=15)
                entry.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "sex":
                var = tk.StringVar(value="Male")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["Male", "Female"], state="readonly", width=13)
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            elif field_type == "yesno":
                var = tk.StringVar(value="no")
                combo = ttk.Combobox(scrollable_frame, textvariable=var, values=["yes", "no"], state="readonly", width=13)
                combo.grid(row=row, column=col*2+1, padx=5, pady=5)
            
            self.input_vars[field_name] = var
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Prediction Button
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        predict_button = ttk.Button(button_frame, text="Predict Diagnosis", command=self.predict_diagnosis)
        predict_button.pack(side=tk.LEFT, padx=10)
        
        clear_button = ttk.Button(button_frame, text="Clear Form", command=self.clear_form)
        clear_button.pack(side=tk.LEFT, padx=10)
        
        # Results
        results_frame = ttk.LabelFrame(self.root, text="Prediction Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Real AI models loaded and trained")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X)
    
    def collect_input_data(self):
        """Collect input data from form"""
        
        input_data = {}
        for field_name, var in self.input_vars.items():
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
        """Make prediction"""
        
        try:
            # Update status
            self.status_var.set("Processing prediction...")
            
            # Collect input data
            input_data = self.collect_input_data()
            
            # Validate input
            if not self.validate_input(input_data):
                self.status_var.set("Validation failed")
                return
            
            # Make prediction using real AI models
            threading.Thread(target=self._make_prediction, args=(input_data,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_var.set("Error")
    
    def _make_prediction(self, input_data):
        """Make prediction using real AI models"""
        
        try:
            # Simulate processing time for better UX
            import time
            time.sleep(1)
            
            # Get selected model
            model_name = self.model_var.get()
            
            # Use real AI predictor
            prediction, prediction_proba = self.predictor.predict(model_name, input_data)
            
            # Format results
            diagnosis = "Appendicitis" if prediction == 1 else "No Appendicitis"
            confidence = prediction_proba[prediction]
            
            # Calculate risk score for display
            risk_score = self._calculate_risk_score(input_data)
            
            results = self._format_real_model_results(input_data, model_name, diagnosis, confidence, risk_score, prediction_proba)
            
            # Update results display
            self.root.after(0, self._display_results, results)
            
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            self.root.after(0, self._display_error, error_msg)
    
    def _calculate_risk_score(self, input_data):
        """Calculate simple risk score for display purposes"""
        
        risk_score = 0
        
        # Age factor
        age = input_data.get('Age', 0)
        if 10 <= age <= 15:
            risk_score += 2
        
        # Temperature factor
        temp = input_data.get('Body_Temperature', 36.5)
        if temp > 37.5:
            risk_score += 2
        elif temp > 38.0:
            risk_score += 3
        
        # WBC Count factor
        wbc = input_data.get('WBC_Count', 0)
        if wbc > 10:
            risk_score += 2
        elif wbc > 15:
            risk_score += 3
        
        # CRP factor
        crp = input_data.get('CRP', 0)
        if crp > 10:
            risk_score += 2
        elif crp > 20:
            risk_score += 3
        
        # Appendix Diameter factor
        diameter = input_data.get('Appendix_Diameter', 0)
        if diameter > 6:
            risk_score += 2
        elif diameter > 8:
            risk_score += 3
        
        # Symptom factors
        if input_data.get('Lower_Right_Abd_Pain') == 'yes':
            risk_score += 2
        if input_data.get('Nausea') == 'yes':
            risk_score += 1
        if input_data.get('Peritonitis') == 'yes':
            risk_score += 3
        
        return risk_score
    
    def _format_real_model_results(self, input_data, model_name, diagnosis, confidence, risk_score, prediction_proba):
        """Format results from real AI models"""
        
        results = f"""
{'='*60}
REAL AI MODEL PREDICTION RESULTS
{'='*60}

Model Used: {model_name}
Model Type: {self.predictor.get_model_info(model_name)['type']}
Trained: {self.predictor.get_model_info(model_name)['trained']}

Diagnosis: {diagnosis}
Confidence: {confidence:.2%}
Risk Score: {risk_score}/20

Prediction Probabilities:
- Appendicitis: {prediction_proba[1]:.2%}
- No Appendicitis: {prediction_proba[0]:.2%}

Input Summary:
- Age: {input_data.get('Age', 'N/A')} years
- Temperature: {input_data.get('Body_Temperature', 'N/A')}°C
- WBC Count: {input_data.get('WBC_Count', 'N/A')} ×10^9/L
- CRP: {input_data.get('CRP', 'N/A')} mg/L
- Appendix Diameter: {input_data.get('Appendix_Diameter', 'N/A')} mm
- Lower Right Pain: {input_data.get('Lower_Right_Abd_Pain', 'N/A')}
- Peritonitis: {input_data.get('Peritonitis', 'N/A')}

Medical Interpretation:
{'='*60}
"""
        
        if diagnosis == "Appendicitis":
            results += """
⚠️  HIGH RISK OF APPENDICITIS DETECTED
Recommendations:
- Immediate medical evaluation recommended
- Consider surgical consultation
- Monitor for worsening symptoms
- Prepare for possible appendectomy

Risk Factors Present:
"""
            temp = input_data.get('Body_Temperature', 36.5)
            wbc = input_data.get('WBC_Count', 0)
            crp = input_data.get('CRP', 0)
            diameter = input_data.get('Appendix_Diameter', 0)
                
            if temp > 37.5: results += f"- Elevated temperature ({temp}°C)\n"
            if wbc > 10: results += f"- Elevated WBC count ({wbc} ×10^9/L)\n"
            if crp > 10: results += f"- Elevated CRP ({crp} mg/L)\n"
            if diameter > 6: results += f"- Enlarged appendix ({diameter} mm)\n"
            if input_data.get('Peritonitis') == 'yes': results += "- Peritonitis present\n"
        
        else:
            results += """
✅ LOW RISK OF APPENDICITIS
Recommendations:
- Continue monitoring
- Consider alternative diagnoses
- Re-evaluate if symptoms worsen
- Follow-up examination recommended

Low Risk Indicators:
"""
            temp = input_data.get('Body_Temperature', 36.5)
            wbc = input_data.get('WBC_Count', 0)
            crp = input_data.get('CRP', 0)
            diameter = input_data.get('Appendix_Diameter', 0)
            
            if temp <= 37.5: results += f"- Normal temperature ({temp}°C)\n"
            if wbc <= 10: results += f"- Normal WBC count ({wbc} ×10^9/L)\n"
            if crp <= 10: results += f"- Normal CRP ({crp} mg/L)\n"
            if diameter <= 6: results += f"- Normal appendix size ({diameter} mm)\n"
            if input_data.get('Peritonitis') == 'no': results += "- No peritonitis\n"
            
            results += f"""
{'='*60}
Note: This prediction uses trained AI models for accurate results.
Clinical judgment should always prevail.
{'='*60}
"""
        
        return results
    
    def _display_results(self, results):
        """Display prediction results"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        self.status_var.set("Real AI prediction complete")
    
    def _display_error(self, error_msg):
        """Display error message"""
        
        messagebox.showerror("Prediction Error", error_msg)
        self.status_var.set("Error")
    
    def clear_form(self):
        """Clear all form fields"""
        
        # Reset demographic fields
        for var in self.input_vars.values():
            if isinstance(var, tk.StringVar):
                if var == self.input_vars.get('Sex'):
                    var.set('Male')
                else:
                    var.set('no')
            else:
                var.set(0.0)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.status_var.set("Form cleared")

def main():
    """Main function to run the GUI application"""
    
    root = tk.Tk()
    app = SimpleAppendicitisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
