import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import ParameterGrid, KFold
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from itertools import product
import json

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AppendicitisDataset(Dataset):
    """Custom dataset for appendicitis prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class FeatureEmbedding(nn.Module):
    """Feature-specific embeddings for tabular data"""
    
    def __init__(self, feature_info, embed_dim=64):
        super(FeatureEmbedding, self).__init__()
        
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        
        numerical_features = [f for f, info in feature_info.items() if info['type'] == 'numerical']
        categorical_features = [f for f, info in feature_info.items() if info['type'] == 'categorical']
        
        self.numerical_embeddings = nn.ModuleDict({
            feat: nn.Linear(1, embed_dim)  # Each numerical feature gets its own embedding
            for feat in numerical_features
        })
        
        # Categorical feature embeddings
        self.categorical_embeddings = nn.ModuleDict({
            feat: nn.Embedding(info['unique_values'], embed_dim)
            for feat, info in feature_info.items() if info['type'] == 'categorical'
        })
        
        # Feature type tokens
        self.numerical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        self.categorical_token = nn.Parameter(torch.randn(1, embed_dim) * 0.1)
        
        # Feature positional encoding
        total_features = len(feature_info)
        self.feature_positional_encoding = nn.Parameter(
            torch.randn(total_features, embed_dim) * 0.1
        )
        
    def forward(self, x_dict):
        """Forward pass with feature-specific embeddings"""
        embeddings = []
        
        # Numerical features (each feature separately)
        if 'numerical' in x_dict:
            numerical_data = x_dict['numerical']  # (batch_size, num_numerical_features)
            num_features_list = list(self.numerical_embeddings.keys())
            
            for i, feat_name in enumerate(num_features_list):
                if i < numerical_data.shape[1]:
                    feat_values = numerical_data[:, i:i+1]  # (batch_size, 1)
                    if feat_name in self.numerical_embeddings:
                        num_embed = self.numerical_embeddings[feat_name](feat_values)
                        num_embed = num_embed.unsqueeze(1)  # (batch_size, 1, embed_dim)
                        num_embed = num_embed + self.numerical_token.unsqueeze(0)
                        embeddings.append(num_embed)
        
        # Categorical features (at top level, not under 'categorical' key)
        for feat_name, feat_values in x_dict.items():
            if feat_name != 'numerical' and feat_name in self.categorical_embeddings:
                cat_embed = self.categorical_embeddings[feat_name](feat_values)
                cat_embed = cat_embed.unsqueeze(1)  # (batch_size, 1, embed_dim)
                cat_embed = cat_embed + self.categorical_token.unsqueeze(0)
                embeddings.append(cat_embed)
        
        # Concatenate all feature embeddings
        if embeddings:
            x = torch.cat(embeddings, dim=1)  # (batch_size, num_features, embed_dim)
            
            # Add feature positional encoding
            seq_len = x.size(1)
            pos_enc = self.feature_positional_encoding[:seq_len].unsqueeze(0)
            x = x + pos_enc
        else:
            # Fallback to simple embedding
            x = torch.zeros(x_dict['numerical'].size(0), 1, self.embed_dim)
        
        return x

class AdvancedTabularTransformer(nn.Module):
    """Advanced Transformer model with proper feature embeddings and multi-feature sequences"""
    
    def __init__(self, feature_info, embed_dim=64, num_heads=8, num_layers=4, 
                 num_classes=2, dropout=0.3):  # Increased dropout for regularization
        super(AdvancedTabularTransformer, self).__init__()
        
        self.feature_info = feature_info
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Feature-specific embeddings
        self.feature_embedding = FeatureEmbedding(feature_info, embed_dim)
        
        # Adaptive number of heads based on embed_dim
        self.num_heads = min(num_heads, embed_dim // 32)
        
        # Transformer encoder layers with proper configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=self.num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        
        # Multiple transformer layers for feature interaction
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head with proper architecture - SINGLE OUTPUT for binary classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(embed_dim // 4, 1)  # Single output for binary classification
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper strategy"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x_dict):
        # Feature-specific embeddings
        x = self.feature_embedding(x_dict)  # (batch_size, num_features, embed_dim)
        
        # Multi-head attention between features
        x = self.transformer(x)  # (batch_size, num_features, embed_dim)
        
        # Simple mean pooling (more stable)
        x_pooled = torch.mean(x, dim=1)  # (batch_size, embed_dim)
        
        # Classification
        output = self.classifier(x_pooled)  # (batch_size, num_classes)
        
        return output

class HyperparameterTuner:
    """Automated hyperparameter tuning for Transformer model"""
    
    def __init__(self, feature_info, device='cpu'):
        self.feature_info = feature_info
        self.device = device
        
        # Define MODEL hyperparameter search space (for model architecture)
        self.param_grid = {
            'embed_dim': [32, 64, 128, 256],
            'num_heads': [4, 8, 16],
            'num_layers': [2, 4, 6, 8],
            'dropout': [0.1, 0.2, 0.3]
        }
        
        # Training parameters (separate from model parameters) - single values for now
        self.training_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        }
        
    def adaptive_architecture(self, num_features):
        """Determine optimal architecture based on feature count"""
        
        # Dynamic embedding dimension based on feature count
        embed_dim = max(64, min(256, num_features * 4))
        
        # Adaptive number of heads
        num_heads = min(8, max(4, embed_dim // 32))
        
        # Adaptive number of layers
        num_layers = min(6, max(2, num_features // 5))
        
        return {
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers
        }
    
    def grid_search(self, X_dict, y, cv_folds=3, max_trials=20):
        """Perform grid search with cross-validation"""
        
        print("Starting hyperparameter tuning...")
        
        # Get adaptive architecture
        num_features = len(self.feature_info)
        adaptive_params = self.adaptive_architecture(num_features)
        
        # Generate parameter combinations (limit to max_trials)
        param_combinations = list(ParameterGrid(self.param_grid))
        if len(param_combinations) > max_trials:
            param_combinations = param_combinations[:max_trials]
        
        best_score = 0
        best_params = None
        results = []
        
        for i, params in enumerate(param_combinations):
            print(f"\nTrial {i+1}/{len(param_combinations)}: {params}")
            
            # Use adaptive parameters as defaults
            final_params = adaptive_params.copy()
            final_params.update(params)
            
            try:
                # Cross-validation with separate model and training parameters
                cv_scores = self._cross_validate(X_dict, y, final_params, self.training_params, cv_folds)
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                result = {
                    'params': final_params,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'cv_scores': cv_scores
                }
                results.append(result)
                
                print(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = final_params
                    print(f"New best score: {best_score:.4f}")
                    
            except Exception as e:
                print(f"Trial failed: {e}")
                continue
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best CV score: {best_score:.4f}")
        
        return best_params, results
    
    def _cross_validate(self, X_dict, y, model_params, training_params, cv_folds):
        """Perform cross-validation with given parameters"""
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(y)):
            print(f"  Fold {fold+1}/{cv_folds}")
            
            # Split data
            X_train_dict = self._split_dict(X_dict, train_idx)
            X_val_dict = self._split_dict(X_dict, val_idx)
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model with model parameters only
            model = AdvancedTabularTransformer(
                feature_info=self.feature_info,
                **model_params
            )
            
            # Create datasets
            train_dataset = AppendicitisDatasetDict(X_train_dict, y_train)
            val_dataset = AppendicitisDatasetDict(X_val_dict, y_val)
            
            # Create data loaders with training parameters
            batch_size = training_params.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model (quick training for CV) with training parameters
            trainer = TransformerTrainer(model, self.device)
            learning_rate = training_params.get('learning_rate', 0.001)
            weight_decay = training_params.get('weight_decay', 1e-4)
            trainer.optimizer = optim.AdamW(
                model.parameters(), 
                lr=learning_rate, 
                weight_decay=weight_decay
            )
            
            # Train for few epochs
            trainer.train(train_loader, val_loader, epochs=5, patience=3)
            
            # Evaluate
            _, val_acc, _, _ = trainer.evaluate(val_loader)
            cv_scores.append(val_acc)
        
        return cv_scores
    
    def _split_dict(self, X_dict, indices):
        """Split dictionary data by indices"""
        result = {}
        for key, value in X_dict.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[indices]
            else:
                result[key] = value[indices]
        return result

class AppendicitisDatasetDict(Dataset):
    """Dataset for dictionary-based features"""
    
    def __init__(self, features_dict, targets):
        self.features_dict = features_dict
        if hasattr(targets, 'values'):
            targets = targets.values
        self.targets = torch.LongTensor(np.array(targets, dtype=np.int64))
        
        # Get batch size from first feature
        first_key = list(features_dict.keys())[0]
        self.batch_size = features_dict[first_key].size(0)
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        item = {}
        for key, value in self.features_dict.items():
            item[key] = value[idx]
        return item, self.targets[idx]
class TransformerTrainer:
    """Training and evaluation pipeline for transformer model"""
    
    def __init__(self, model, device='cpu', pos_weight=None):
        self.model = model.to(device)
        self.device = device
        # BCEWithLogitsLoss for single-output binary classification
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device))
            print(f"[BCEWithLogitsLoss] pos_weight={pos_weight:.4f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            print("[BCEWithLogitsLoss] no class weighting")
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.temperature = 1.0  # Will be calibrated after training
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features_dict, targets in dataloader:
            targets = targets.to(self.device)
            features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
            
            self.optimizer.zero_grad()
            logits = self.model(features_dict).squeeze(-1)  # (batch_size,)
            targets_float = targets.float()  # BCE requires float targets
            loss = self.criterion(logits, targets_float)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).long()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        return total_loss / len(dataloader), 100 * correct / total
    
    def evaluate(self, dataloader, debug=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probs = []
        all_logits = []
        
        with torch.no_grad():
            for features_dict, targets in dataloader:
                targets = targets.to(self.device)
                features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
                
                logits = self.model(features_dict).squeeze(-1)  # (batch_size,)
                targets_float = targets.float()
                loss = self.criterion(logits, targets_float)
                
                total_loss += loss.item()
                probs = torch.sigmoid(logits / self.temperature)  # Apply temperature scaling
                predicted = (probs > 0.5).long()
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        if debug:
            logits_arr = np.array(all_logits)
            probs_arr = np.array(all_probs)
            print(f"  [DEBUG] Logits - min:{logits_arr.min():.4f} max:{logits_arr.max():.4f} "
                  f"mean:{logits_arr.mean():.4f} std:{logits_arr.std():.4f}")
            print(f"  [DEBUG] Probs  - min:{probs_arr.min():.4f} max:{probs_arr.max():.4f} "
                  f"mean:{probs_arr.mean():.4f} std:{probs_arr.std():.4f}")
        
        return total_loss / len(dataloader), 100 * correct / total, all_predictions, all_targets
    
    def calibrate_temperature(self, val_loader):
        """Calibrate temperature scaling on validation set for better probability estimates"""
        self.model.eval()
        nll_criterion = nn.BCEWithLogitsLoss()
        
        # Collect all logits and targets
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for features_dict, targets in val_loader:
                targets = targets.to(self.device)
                features_dict = {k: v.to(self.device) for k, v in features_dict.items()}
                logits = self.model(features_dict).squeeze(-1)
                all_logits.append(logits)
                all_targets.append(targets.float())
        
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        
        # Optimize temperature
        temperature = torch.tensor([1.0], requires_grad=True, device=self.device)
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = all_logits / temperature
            loss = nll_criterion(scaled_logits, all_targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.temperature = temperature.item()
        print(f"[CALIBRATION] Temperature scaled: {self.temperature:.4f}")
        return self.temperature

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader, debug=(epoch % 10 == 0))
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping on validation loss (more stable than accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state dict
                torch.save(self.model.state_dict(), 'best_advanced_transformer_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Calibrate temperature after training
        print("\n[CALIBRATION] Running temperature scaling...")
        self.calibrate_temperature(val_loader)
        
        return training_history

def analyze_features(X):
    """Analyze feature types and determine optimal dimensions"""
    
    feature_info = {}
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            # Numerical feature
            unique_vals = len(X[col].unique())
            feature_info[col] = {
                'type': 'numerical',
                'unique_values': unique_vals,
                'embed_dim': 8,  # Base dimension for numerical
                'range': [float(X[col].min()), float(X[col].max())],
                'mean': float(X[col].mean()),
                'std': float(X[col].std())
            }
        else:
            # Categorical feature
            unique_vals = len(X[col].unique())
            feature_info[col] = {
                'type': 'categorical',
                'unique_values': unique_vals,
                'embed_dim': min(16, max(4, unique_vals // 2)),
                'categories': list(X[col].unique())
            }
    
    print("\n=== FEATURE ANALYSIS ===")
    print(f"Total features: {len(feature_info)}")
    
    numerical_features = [f for f, info in feature_info.items() if info['type'] == 'numerical']
    categorical_features = [f for f, info in feature_info.items() if info['type'] == 'categorical']
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    print("\nNumerical Features:")
    for feat in numerical_features:
        info = feature_info[feat]
        print(f"  {feat}: range [{info['range'][0]:.2f}, {info['range'][1]:.2f}], mean={info['mean']:.2f}")
    
    print("\nCategorical Features:")
    for feat in categorical_features:
        info = feature_info[feat]
        print(f"  {feat}: {info['unique_values']} categories - {info['categories']}")
    
    return feature_info

def prepare_data_for_advanced_transformer(X, y, feature_info):
    """Prepare data for advanced transformer with proper feature separation"""
    
    print("\n=== DATA PREPARATION FOR ADVANCED TRANSFORMER ===")
    
    # Load original dataset if X is already preprocessed
    if X.shape[1] > 50:  # Likely one-hot encoded
        print("Detected one-hot encoded data, loading original dataset...")
        try:
            # Load original dataset
            original_df = pd.read_excel('../Dataset 1/app_data.xlsx')
            print(f"Original dataset shape: {original_df.shape}")
            
            # Extract target
            if 'Management' in original_df.columns:
                y_original = original_df['Management'].values
            elif 'Diagnosis' in original_df.columns:
                y_original = original_df['Diagnosis'].values
            else:
                # Find target column
                target_cols = [col for col in original_df.columns if col.lower() in ['management', 'diagnosis', 'target']]
                if target_cols:
                    y_original = original_df[target_cols[0]].values
                else:
                    raise ValueError("No target column found")
            
            # Define feature columns (exclude target and auxiliary columns)
            exclude_cols = ['Management', 'Diagnosis_Presumptive', 'Diagnosis', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Length_of_Stay']
            feature_columns = [col for col in original_df.columns if col not in exclude_cols]
            X_original = original_df[feature_columns].copy()
            
            print(f"Using original features: {len(feature_columns)}")
            print(f"Feature columns: {feature_columns[:15]}...")  # First 15
            
        except Exception as e:
            print(f"Could not load original dataset: {e}")
            print("Using provided preprocessed data...")
            X_original = X
            y_original = y.values.flatten() if hasattr(y, 'values') else y
    else:
        X_original = X
        y_original = y.values.flatten() if hasattr(y, 'values') else y
    
    # Clean the data
    print("Cleaning original dataset...")
    
    # Remove rows with missing target
    valid_indices = ~pd.isna(y_original)
    X_clean = X_original[valid_indices]
    y_clean = y_original[valid_indices]
    
    # Ensure consistent length
    min_len = min(len(X_clean), len(y_clean))
    X_clean = X_clean.iloc[:min_len]
    y_clean = y_clean[:min_len]
    
    # Convert to numpy arrays
    if hasattr(X_clean, 'values'):
        X_array = X_clean.values
    else:
        X_array = X_clean
    
    # Handle mixed data types - only convert numerical columns to float
    numeric_cols = []
    for i, col in enumerate(X_clean.columns if hasattr(X_clean, 'columns') else range(X_array.shape[1])):
        if hasattr(X_clean, 'columns'):
            col_data = X_clean[col]
            if col_data.dtype in ['int64', 'float64']:
                numeric_cols.append(X_array[:, i] if hasattr(X_array, 'shape') else col_data.values)
            else:
                # Keep categorical as object for now
                pass
        else:
            # If no column names, assume all are numerical
            numeric_cols.append(X_array[:, i] if hasattr(X_array, 'shape') else X_clean)
    
    # Stack numerical columns
    if numeric_cols:
        X_array_numeric = np.column_stack(numeric_cols).astype(np.float32)
    else:
        X_array_numeric = X_array.astype(np.float32) if hasattr(X_array, 'astype') else np.array(X_clean, dtype=np.float32)
    
    y_array = y_clean.flatten()
    
    # Encode target variable - ensure binary classification
    label_encoder = LabelEncoder()
    
    # Convert to binary: appendicitis vs no appendicitis
    # Management has: conservative, primary surgical, secondary surgical, simultaneous appendectomy
    # We'll consider surgical cases as appendicitis (1) and conservative as no appendicitis (0)
    y_binary = np.array([1 if str(x).lower() in ['primary surgical', 'secondary surgical', 'simultaneous appendectomy'] else 0 for x in y_array])
    
    y_encoded = label_encoder.fit_transform(y_binary)
    
    print(f"Data after cleaning: {X_array.shape}")
    print(f"Target distribution: {np.bincount(y_encoded)}")
    print(f"Target classes: {label_encoder.classes_}")
    
    # Create feature dictionary for advanced transformer
    X_dict = {}
    
    # Define explicit feature types based on medical knowledge
    numerical_features = [
        'Age', 'Weight', 'Height', 'BMI', 'Body_Temperature', 
        'WBC_Count', 'RBC_Count', 'Hemoglobin', 'RDW', 
        'Segmented_Neutrophils', 'Thrombocyte_Count', 'CRP', 'Neutrophil_Percentage'
    ]
    
    categorical_features = [
        'Sex', 'Lower_Right_Abd_Pain', 'Migratory_Pain', 'Loss_of_Appetite', 
        'Nausea', 'Coughing_Pain', 'Dysuria', 'Stool', 'Peritonitis', 
        'Severity', 'Contralateral_Rebound_Tenderness', 'Ipsilateral_Rebound_Tenderness', 
        'Psoas_Sign', 'Neutrophilia', 'Ketones_in_Urine', 'RBC_in_Urine', 'WBC_in_Urine'
    ]
    
    # Process numerical features
    numerical_data = []
    numerical_feature_names = []
    
    # Get numerical features from our predefined list
    for feat in numerical_features:
        if hasattr(X_clean, 'columns') and feat in X_clean.columns:
            # Check if column is actually numerical
            if X_clean[feat].dtype in ['int64', 'float64']:
                # Normalize numerical feature with NaN handling
                values = X_clean[feat].values
                
                # Replace NaN with mean
                if np.isnan(values).any():
                    mean_val = np.nanmean(values)
                    values = np.where(np.isnan(values), mean_val, values)
                    print(f"  {feat}: filled {np.isnan(X_clean[feat].values).sum()} NaN values with mean {mean_val:.2f}")
                
                values_norm = (values - np.mean(values)) / (np.std(values) + 1e-8)
                numerical_data.append(values_norm)
                numerical_feature_names.append(feat)
                print(f"  {feat}: {X_clean[feat].dtype} - numerical")
            else:
                print(f"  {feat}: {X_clean[feat].dtype} - skipping (not numerical)")
    
    # Process categorical features
    categorical_data = {}
    categorical_encoders = {}
    
    for feat in categorical_features:
        if hasattr(X_clean, 'columns') and feat in X_clean.columns:
            # Check if column is actually categorical
            if X_clean[feat].dtype == 'object':
                # Encode categorical feature
                values = X_clean[feat].values.astype(str)
                le = LabelEncoder()
                cat_encoded = le.fit_transform(values)
                categorical_data[feat] = cat_encoded
                categorical_encoders[feat] = le
                print(f"  {feat}: {X_clean[feat].dtype} - categorical")
            else:
                print(f"  {feat}: {X_clean[feat].dtype} - skipping (not categorical)")
    
    # If no features found, use first N columns as fallback
    if len(numerical_data) == 0 and hasattr(X_array, 'shape'):
        # Use first 13 columns as numerical
        for i in range(min(13, X_array_numeric.shape[1])):
            numerical_data.append(X_array_numeric[:, i])
            numerical_feature_names.append(f'num_feature_{i}')
    
    if len(categorical_data) == 0 and hasattr(X_array, 'shape'):
        # Use remaining columns as categorical
        start_idx = len(numerical_data)
        for i in range(min(13, X_array_numeric.shape[1] - start_idx)):
            col_name = f'cat_feature_{i}'
            values = X_array_numeric[:, start_idx + i].astype(str)
            le = LabelEncoder()
            cat_encoded = le.fit_transform(values)
            categorical_data[col_name] = cat_encoded
            categorical_encoders[col_name] = le
    
    # Convert to tensors
    if numerical_data:
        X_dict['numerical'] = torch.FloatTensor(np.column_stack(numerical_data))
    
    for col, values in categorical_data.items():
        X_dict[col] = torch.LongTensor(values)
    
    print(f"Feature dictionary created with keys: {list(X_dict.keys())}")
    
    if 'numerical' in X_dict:
        print(f"Numerical features shape: {X_dict['numerical'].shape}")
        print(f"Numerical features: {numerical_feature_names}")
    
    for col in categorical_data:
        print(f"Categorical '{col}' shape: {X_dict[col].shape}")
    
    # Update feature_info to match actual data
    actual_feature_info = {}
    
    # Add numerical features
    for i, feat in enumerate(numerical_feature_names):
        actual_feature_info[feat] = {
            'type': 'numerical',
            'unique_values': len(X_dict['numerical']) if 'numerical' in X_dict else 0,
            'embed_dim': 8,
            'range': [float(np.min(numerical_data[i])), float(np.max(numerical_data[i]))],
            'mean': float(np.mean(numerical_data[i])),
            'std': float(np.std(numerical_data[i]))
        }
    
    # Add categorical features
    for feat, values in categorical_data.items():
        actual_feature_info[feat] = {
            'type': 'categorical',
            'unique_values': len(np.unique(values)),
            'embed_dim': min(16, max(4, len(np.unique(values)) // 2)),
            'categories': list(np.unique(values))
        }
    
    print(f"Actual feature info created with {len(actual_feature_info)} features")
    print(f"Numerical: {len([f for f, info in actual_feature_info.items() if info['type'] == 'numerical'])}")
    print(f"Categorical: {len([f for f, info in actual_feature_info.items() if info['type'] == 'categorical'])}")
    
    return X_dict, y_encoded, label_encoder, categorical_encoders, actual_feature_info

def calculate_medical_metrics(y_true, y_pred):
    """Calculate medical-specific evaluation metrics"""
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    sensitivity = recall_score(y_true, y_pred, average='binary')  # Sensitivity = Recall
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate PPV and NPV
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def load_preprocessed_data():
    """Load preprocessed data from unified preprocessing module"""
    
    import sys
    import os
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from unified_data_preprocessing import prepare_unified_data
    
    # Load unified data (same as all baseline models)
    X_train, X_test, y_train, y_test, feature_names, encoders, scalers = prepare_unified_data('CSV')
    
    datasets = [('CSV', X_train, X_test, y_train, y_test, feature_names, encoders, scalers)]
    
    return datasets

def prepare_data_for_transformer(X, y):
    """Prepare data for transformer model"""
    
    # Convert to numpy arrays first
    X_array = X.values.astype(np.float32)
    y_array = y.values.flatten()
    
    # Ensure both arrays have the same length
    min_length = min(len(X_array), len(y_array))
    X_array = X_array[:min_length]
    y_array = y_array[:min_length]
    
    # Find indices where target is not NaN
    valid_indices = ~pd.isna(y_array)
    
    # Filter both features and targets
    X_clean = X_array[valid_indices]
    y_clean = y_array[valid_indices]
    
    # Encode target variable - ensure binary classification
    label_encoder = LabelEncoder()
    
    # Convert to binary: appendicitis vs no appendicitis
    y_binary = np.array([1 if str(x).lower() == 'appendicitis' else 0 for x in y_clean])
    
    y_encoded = label_encoder.fit_transform(y_binary)
    
    print(f"Data after cleaning: {X_clean.shape}")
    print(f"Target distribution: {np.bincount(y_encoded)}")
    print(f"Target classes: {label_encoder.classes_}")
    
    return X_clean, y_encoded, label_encoder

def split_data(X, y, test_size=0.4, random_state=SEED):
    """Split data into training and testing sets (60:40 ratio as per paper)"""
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Testing target distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    """Main advanced transformer model training function with hyperparameter tuning"""
    
    print("=" * 80)
    print("ADVANCED TRANSFORMER MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("With Feature-Specific Embeddings & Hyperparameter Tuning")
    print("=" * 80)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load unified data (same as all baseline models)
        print(f"\n{'='*60}")
        print("LOADING UNIFIED DATA")
        print(f"[FAIR COMPARISON - UNIFIED 30 FEATURES]")
        print(f"{'='*60}")
        datasets = load_preprocessed_data()
        
        for dataset_name, X_train, X_test, y_train, y_test, feature_names, encoders, scalers in datasets:
            print(f"\n{'='*80}")
            print(f"PROCESSING {dataset_name.upper()} DATASET")
            print(f"[FAIR COMPARISON]")
            print(f"{'='*80}")
            
            # Step 1: Convert unified data to Transformer format
            print(f"\nConverting unified data to Transformer format...")
            
            # Convert train data to dictionary format
            X_train_dict = {}
            X_test_dict = {}
            
            # Separate numerical and categorical features
            import sys
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from unified_data_preprocessing import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
            
            # Numerical features
            X_train_dict['numerical'] = X_train[NUMERICAL_FEATURES].values.astype(np.float32)
            X_test_dict['numerical'] = X_test[NUMERICAL_FEATURES].values.astype(np.float32)
            
            # Categorical features
            for feat in CATEGORICAL_FEATURES:
                X_train_dict[feat] = torch.from_numpy(X_train[feat].values).long()
                X_test_dict[feat] = torch.from_numpy(X_test[feat].values).long()
            
            # Convert numerical to tensor
            X_train_dict['numerical'] = torch.from_numpy(X_train_dict['numerical']).float()
            X_test_dict['numerical'] = torch.from_numpy(X_test_dict['numerical']).float()
            
            print(f"Train data shape: {X_train_dict['numerical'].shape}")
            print(f"Test data shape: {X_test_dict['numerical'].shape}")
            print(f"Numerical features: {len(NUMERICAL_FEATURES)}")
            print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
            
            # Create feature info for Transformer
            feature_info = {}
            for feat in NUMERICAL_FEATURES:
                feature_info[feat] = {
                    'type': 'numerical',
                    'unique_values': len(X_train[feat].unique()),
                    'embed_dim': 8,
                    'range': [float(X_train[feat].min()), float(X_train[feat].max())],
                    'mean': float(X_train[feat].mean()),
                    'std': float(X_train[feat].std())
                }
            
            for feat in CATEGORICAL_FEATURES:
                feature_info[feat] = {
                    'type': 'categorical',
                    'unique_values': len(X_train[feat].unique()),
                    'embed_dim': min(16, max(4, len(X_train[feat].unique()) // 2)),
                    'categories': list(X_train[feat].unique())
                }
            
            print(f"Feature info created with {len(feature_info)} features")
            
            # Create label encoder for consistency
            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)
            categorical_encoders = encoders
            
            # Step 4: Use default parameters for fair comparison (baseline models already tuned)
            print(f"\n{'='*60}")
            print("USING DEFAULT PARAMETERS FOR FAIR COMPARISON")
            print(f"[Baseline models already have hyperparameter tuning]")
            print(f"{'='*60}")
            
            # Compute class weights for BCEWithLogitsLoss
            y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
            n_negative = int(np.sum(y_train_arr == 0))
            n_positive = int(np.sum(y_train_arr == 1))
            # pos_weight weights the positive class; cap to avoid extreme values
            raw_pw = n_negative / max(n_positive, 1)
            pos_weight = max(0.05, min(raw_pw, 4.0))
            print(f"Class distribution - No Appendicitis: {n_negative}, Appendicitis: {n_positive}")
            print(f"pos_weight for BCEWithLogitsLoss: {pos_weight:.4f} (raw={raw_pw:.4f})")

            model_params = {
                'embed_dim': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.2,
                'num_classes': 2
            }
            training_params = {
                'batch_size': 32,
                'learning_rate': 0.0005,
                'weight_decay': 1e-3
            }
            tuning_results = []
            
            print(f"Model parameters: {model_params}")
            print(f"Training parameters: {training_params}")
            
            # Step 5: Initialize model with best parameters
            print(f"\n{'='*60}")
            print("INITIALIZING MODEL WITH BEST PARAMETERS")
            print(f"{'='*60}")
            
            try:
                model = AdvancedTabularTransformer(
                    feature_info=feature_info,
                    **model_params
                )
            except Exception as e:
                print(f"Error initializing model: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"Model initialized with parameters: {model_params}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Create datasets
            train_dataset = AppendicitisDatasetDict(X_train_dict, y_train)
            test_dataset = AppendicitisDatasetDict(X_test_dict, y_test)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False)
            
            # Initialize trainer with BCEWithLogitsLoss and pos_weight
            trainer = TransformerTrainer(model, device, pos_weight=pos_weight)
            trainer.optimizer = optim.AdamW(
                model.parameters(), 
                lr=training_params['learning_rate'], 
                weight_decay=training_params['weight_decay']
            )
            
            # Train model (more epochs for better convergence)
            print(f"\nTraining advanced transformer model...")
            training_history = trainer.train(train_loader, test_loader, epochs=100, patience=15)
            
            # Load best model for evaluation
            model.load_state_dict(torch.load('best_advanced_transformer_model.pth', weights_only=True))
            
            # Final evaluation with debug output
            print(f"\n[FINAL EVALUATION]")
            test_loss, test_acc, predictions, targets = trainer.evaluate(test_loader, debug=True)
            
            # Calculate medical metrics
            metrics = calculate_medical_metrics(targets, predictions)
            
            print(f"\n{'='*80}")
            print(f"FINAL RESULTS - {dataset_name.upper()} DATASET")
            print(f"{'='*80}")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"\nMedical Metrics:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Sensitivity (Recall): {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  PPV (Positive Predictive Value): {metrics['ppv']:.4f}")
            print(f"  NPV (Negative Predictive Value): {metrics['npv']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  True Positives: {metrics['tp']}")
            print(f"  True Negatives: {metrics['tn']}")
            print(f"  False Positives: {metrics['fp']}")
            print(f"  False Negatives: {metrics['fn']}")
            
            print(f"\nModel Architecture:")
            print(f"  Embedding Dimension: {model_params['embed_dim']}")
            print(f"  Number of Heads: {model.num_heads}")
            print(f"  Number of Layers: {model_params['num_layers']}")
            print(f"  Dropout: {model_params['dropout']}")
            print(f"  Learning Rate: {training_params['learning_rate']}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"advanced_transformer_results_fair_{dataset_name}_{timestamp}.pkl"
            
            results = {
                'dataset_name': dataset_name,
                'comparison_type': 'fair',
                'data_type': 'unified_30_features',
                'model_architecture': str(model),
                'feature_info': feature_info,
                'model_parameters': model_params,
                'training_parameters': training_params,
                'tuning_results': tuning_results,
                'training_history': training_history,
                'final_metrics': metrics,
                'label_encoder': label_encoder,
                'categorical_encoders': categorical_encoders,
                'input_features': feature_names,
                'timestamp': timestamp,
                'temperature': trainer.temperature,
                'pos_weight': pos_weight
            }
            
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"\nResults saved to: {results_file}")
            
            # Save full checkpoint for GUI inference (.pt)
            gui_checkpoint_path = Path(__file__).parent.parent / '09_GUI_Application' / 'saved_models' / 'Transformer.pt'
            gui_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use StandardScaler params from metadata (computed on RAW data before scaling)
            # This ensures GUI can normalize raw input the same way training data was normalized
            scaler = scalers  # This is the numerical_scalers from prepare_unified_data
            norm_means = scaler.mean_.astype(np.float32) if hasattr(scaler, 'mean_') else np.zeros(len(NUMERICAL_FEATURES), dtype=np.float32)
            norm_stds = scaler.scale_.astype(np.float32) if hasattr(scaler, 'scale_') else np.ones(len(NUMERICAL_FEATURES), dtype=np.float32)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_info': feature_info,
                'embed_dim': model_params['embed_dim'],
                'num_heads': model_params['num_heads'],
                'num_layers': model_params['num_layers'],
                'dropout': model_params['dropout'],
                'is_trained': True,
                'normalization_means': norm_means,
                'normalization_stds': norm_stds,
                'temperature': trainer.temperature,
                'pos_weight': pos_weight
            }, gui_checkpoint_path)
            print(f"GUI checkpoint saved to: {gui_checkpoint_path}")
        
        print(f"\n{'='*80}")
        print("ADVANCED TRANSFORMER MODEL TRAINING COMPLETED")
        print("[OK] Feature-specific embeddings implemented")
        print("[OK] Multi-feature sequence processing")
        print("[OK] Model training completed")
        print("[OK] Proper Transformer architecture")
        print("[OK] BCEWithLogitsLoss with class weighting")
        print("[OK] Temperature scaling calibrated")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

def split_data_for_dict(X_dict, y, test_size=0.4, random_state=SEED):
    """Split dictionary data into training and testing sets"""
    
    from sklearn.model_selection import train_test_split
    
    # Get indices for splitting
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split dictionary data
    X_train_dict = {}
    X_test_dict = {}
    
    for key, value in X_dict.items():
        X_train_dict[key] = value[train_indices]
        X_test_dict[key] = value[test_indices]
    
    # Split targets
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training set: {len(y_train)} samples")
    print(f"Testing set: {len(y_test)} samples")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Testing target distribution: {np.bincount(y_test)}")
    
    return X_train_dict, X_test_dict, y_train, y_test

class TransformerModel:
    """Real PyTorch Transformer model with sklearn-compatible interface"""
    
    def __init__(self, embed_dim=64, num_heads=8, num_layers=4, dropout=0.1):
        print("REAL TRANSFORMER MODEL (PyTorch) INITIALIZED")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.feature_info = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_feature_info(self, X):
        """Create feature information for tabular data"""
        feature_info = {}
        if hasattr(X, 'columns'):
            for i, col in enumerate(X.columns):
                feature_info[col] = {
                    'type': 'numerical',
                    'unique_values': len(X[col].unique()),
                    'embed_dim': 8,
                    'range': [float(X[col].min()), float(X[col].max())],
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std())
                }
        else:
            # For numpy arrays
            for i in range(X.shape[1]):
                feature_info[f'feature_{i}'] = {
                    'type': 'numerical',
                    'unique_values': len(np.unique(X[:, i])),
                    'embed_dim': 8,
                    'range': [float(np.min(X[:, i])), float(np.max(X[:, i]))],
                    'mean': float(np.mean(X[:, i])),
                    'std': float(np.std(X[:, i]))
                }
        return feature_info
    
    def train_model(self, X_train, y_train):
        """Train the PyTorch Transformer model"""
        print("Training Transformer with PyTorch...")
        
        # Create feature info
        self.feature_info = self._create_feature_info(X_train)
        
        # Create model
        self.model = AdvancedTabularTransformer(
            feature_info=self.feature_info,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=2
        )
        
        self.model = self.model.to(self.device)
        
        # Convert data to tensors
        if hasattr(X_train, 'values'):
            X_array = X_train.values
        else:
            X_array = X_train
            
        if hasattr(y_train, 'values'):
            y_array = y_train.values
        else:
            y_array = y_train
        
        # NORMALIZE INPUTS - STEP 1 FIX
        # Replace missing values (zeros) with mean values
        X_normalized = X_array.copy()
        for i in range(X_normalized.shape[1]):
            col_values = X_normalized[:, i]
            # Replace zeros with mean of non-zero values
            non_zero_mask = col_values != 0
            if np.any(non_zero_mask):
                mean_val = np.mean(col_values[non_zero_mask])
                X_normalized[~non_zero_mask, i] = mean_val
        
        # Apply mean normalization
        for i in range(X_normalized.shape[1]):
            col_mean = np.mean(X_normalized[:, i])
            col_std = np.std(X_normalized[:, i])
            if col_std > 0:
                X_normalized[:, i] = (X_normalized[:, i] - col_mean) / col_std
        
        # Create simple dataset (convert to proper format)
        X_tensor = torch.FloatTensor(X_normalized)
        y_tensor = torch.LongTensor(y_array)
        
        # Store normalization parameters for prediction
        self.normalization_means = np.mean(X_array, axis=0)
        self.normalization_stds = np.std(X_array, axis=0)
        self.normalization_means[np.isnan(self.normalization_means)] = 0
        self.normalization_stds[np.isnan(self.normalization_stds)] = 1
        
        # Create simple dataset that works with tensors
        class SimpleDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            
            def __len__(self):
                return len(self.X)
            
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]
        
        dataset = SimpleDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training setup with CLASS BALANCING - STEP 3 FIX
        # Calculate class weights for balancing
        positive_samples = np.sum(y_array == 1)
        negative_samples = np.sum(y_array == 0)
        pos_weight = negative_samples / positive_samples if positive_samples > 0 else 1.0
        print(f"Class balance - Positive: {positive_samples}, Negative: {negative_samples}, Weight: {pos_weight:.2f}")
        
        # Use BCEWithLogitsLoss with class weighting - FIXED for single output
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)  # Lower learning rate
        
        # Training loop with IMPROVEMENTS - STEP 4 FIX
        self.model.train()
        epochs = 50  # Increase epochs for better training
        
        for epoch in range(epochs):
            total_loss = 0
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Convert to dictionary format for the model
                features_dict = {'numerical': features}
                
                optimizer.zero_grad()
                outputs = self.model(features_dict)
                
                # For BCEWithLogitsLoss with single output, targets need to be float
                targets_float = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)
                
                loss = criterion(outputs, targets_float)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_trained = True
        print("REAL TRANSFORMER MODEL TRAINING COMPLETED")
        return self.model
    
    def predict(self, X):
        """Make predictions (0 or 1) with ADJUSTED THRESHOLD - STEP 7 FIX"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
            
            # NORMALIZE INPUT using stored parameters
            X_normalized = X_array.copy()
            for i in range(X_normalized.shape[1]):
                # Replace zeros with mean
                zero_mask = X_normalized[:, i] == 0
                if np.any(zero_mask):
                    X_normalized[zero_mask, i] = self.normalization_means[i]
                
                # Apply normalization
                if self.normalization_stds[i] > 0:
                    X_normalized[:, i] = (X_normalized[:, i] - self.normalization_means[i]) / self.normalization_stds[i]
            
            X_dict = {'numerical': torch.FloatTensor(X_normalized).to(self.device)}
            
            outputs = self.model(X_dict)
            probabilities = torch.sigmoid(outputs.squeeze())  # Apply sigmoid to single logit output
            
            # ADJUSTED DECISION THRESHOLD - STEP 7 FIX
            threshold = 0.5  # Standard threshold for balanced predictions
            predicted = (probabilities > threshold).long()
            
            # DEBUG OUTPUT - STEP 8 FIX
            prob_val = probabilities.cpu().numpy()
            pred_val = predicted.cpu().numpy()
            if prob_val.ndim == 0:
                prob_val = float(prob_val)
                pred_val = int(pred_val)
            else:
                prob_val = prob_val[0]
                pred_val = pred_val[0]
            print(f"TRANSFORMER PROBABILITY: {prob_val:.3f}, THRESHOLD: {threshold}, PREDICTION: {pred_val}")
            
            # Ensure return is always array
            pred_array = predicted.cpu().numpy()
            if pred_array.ndim == 0:
                return np.array([int(pred_array)])
            return pred_array
    
    def predict_proba(self, X):
        """Return probabilities [P(0), P(1)] with PROPER SIGMOID - STEP 5 FIX"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
            
            # NORMALIZE INPUT using stored parameters
            X_normalized = X_array.copy()
            for i in range(X_normalized.shape[1]):
                # Replace zeros with mean
                zero_mask = X_normalized[:, i] == 0
                if np.any(zero_mask):
                    X_normalized[zero_mask, i] = self.normalization_means[i]
                
                # Apply normalization
                if self.normalization_stds[i] > 0:
                    X_normalized[:, i] = (X_normalized[:, i] - self.normalization_means[i]) / self.normalization_stds[i]
            
            X_dict = {'numerical': torch.FloatTensor(X_normalized).to(self.device)}
            
            outputs = self.model(X_dict)
            # Apply sigmoid properly for binary classification
            probabilities = torch.sigmoid(outputs)
            
            # Format as [P(0), P(1)]
            prob_class_0 = (1 - probabilities[:, 0]).cpu().numpy()
            prob_class_1 = probabilities[:, 0].cpu().numpy()
            
            result = np.column_stack([prob_class_0, prob_class_1])
            
            return result
    
    def save(self, filepath):
        """Save model state dict with normalization parameters"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'feature_info': self.feature_info,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'is_trained': self.is_trained,
                'normalization_means': self.normalization_means,
                'normalization_stds': self.normalization_stds
            }, filepath)
            print(f"REAL TRANSFORMER MODEL saved to {filepath}")
        else:
            raise ValueError("No model to save")
    
    def load(self, filepath):
        """Load model state dict with normalization parameters"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Reconstruct feature info
        self.feature_info = checkpoint['feature_info']
        self.embed_dim = checkpoint['embed_dim']
        self.num_heads = checkpoint['num_heads']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.is_trained = checkpoint['is_trained']
        
        # Load normalization parameters
        self.normalization_means = checkpoint['normalization_means']
        self.normalization_stds = checkpoint['normalization_stds']
        
        # Reconstruct model
        self.model = AdvancedTabularTransformer(
            feature_info=self.feature_info,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=2
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        print(f"REAL TRANSFORMER MODEL loaded from {filepath}")

if __name__ == "__main__":
    main()
