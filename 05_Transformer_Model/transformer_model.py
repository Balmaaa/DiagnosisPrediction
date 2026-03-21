import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AppendicitisDataset(Dataset):
    """Custom dataset for appendicitis prediction"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class TabularTransformer(nn.Module):
    """Transformer model for tabular medical data"""
    
    def __init__(self, input_dim, embed_dim=64, num_heads=8, num_layers=4, num_classes=2, dropout=0.1):
        super(TabularTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding for tabular data
        self.positional_encoding = nn.Parameter(torch.randn(1000, embed_dim) * 0.1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)  # (batch_size, embed_dim)
        
        # Add positional encoding - fix dimension mismatch
        batch_size = x.size(0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        seq_len = x.size(1)
        # Use only the needed portion of positional encoding
        pos_enc = self.positional_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)  # (batch_size, seq_len, embed_dim)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        output = self.classifier(x)  # (batch_size, num_classes)
        
        return output

class TransformerTrainer:
    """Training and evaluation pipeline for transformer model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, targets in dataloader:
            features, targets = features.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        return total_loss / len(dataloader), 100 * correct / total
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return total_loss / len(dataloader), 100 * correct / total, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_acc = 0
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.evaluate(val_loader)
            
            self.scheduler.step(val_loss)
            
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
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return training_history

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
    """Load preprocessed data from previous pipeline"""
    
    # Load preprocessed features and targets
    script_dir = Path(__file__).parent
    
    # Try to load from preprocessing pipeline output
    excel_x_path = script_dir.parent / "04_Preprocessing Pipeline" / "excel_preprocessed_features.csv"
    excel_y_path = script_dir.parent / "04_Preprocessing Pipeline" / "excel_target.csv"
    csv_x_path = script_dir.parent / "04_Preprocessing Pipeline" / "csv_preprocessed_features.csv"
    csv_y_path = script_dir.parent / "04_Preprocessing Pipeline" / "csv_target.csv"
    
    datasets = []
    
    # Load Excel dataset if available
    if excel_x_path.exists() and excel_y_path.exists():
        excel_X = pd.read_csv(excel_x_path)
        excel_y = pd.read_csv(excel_y_path, header=None)  # No header for target
        excel_y.columns = ['Diagnosis']  # Add column name
        datasets.append(('Excel', excel_X, excel_y))
        print(f"Loaded Excel dataset: {excel_X.shape}, Target: {excel_y.shape}")
    
    # Load CSV dataset if available
    if csv_x_path.exists() and csv_y_path.exists():
        csv_X = pd.read_csv(csv_x_path)
        csv_y = pd.read_csv(csv_y_path, header=None)  # No header for target
        csv_y.columns = ['Diagnosis']  # Add column name
        datasets.append(('CSV', csv_X, csv_y))
        print(f"Loaded CSV dataset: {csv_X.shape}, Target: {csv_y.shape}")
    
    if not datasets:
        raise FileNotFoundError("No preprocessed datasets found. Run preprocessing pipeline first.")
    
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

def split_data(X, y, test_size=0.4, random_state=42):
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
    """Main transformer model training function"""
    
    print("=" * 60)
    print("TRANSFORMER MODEL FOR PEDIATRIC APPENDICITIS PREDICTION")
    print("=" * 60)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load preprocessed datasets
        datasets = load_preprocessed_data()
        
        for dataset_name, X, y in datasets:
            print(f"\n{'='*60}")
            print(f"TRAINING {dataset_name.upper()} DATASET")
            print(f"{'='*60}")
            
            # Prepare data
            X_array, y_array, label_encoder = prepare_data_for_transformer(X, y)
            
            # Split data (60:40 as specified in paper)
            X_train, X_test, y_train, y_test = split_data(X_array, y_array)
            
            # Create datasets
            train_dataset = AppendicitisDataset(X_train, y_train)
            test_dataset = AppendicitisDataset(X_test, y_test)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Initialize transformer model
            input_dim = X_train.shape[1]
            model = TabularTransformer(
                input_dim=input_dim,
                embed_dim=64,
                num_heads=8,
                num_layers=4,
                num_classes=2,
                dropout=0.1
            )
            
            print(f"Model initialized with input dimension: {input_dim}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Initialize trainer
            trainer = TransformerTrainer(model, device)
            
            # Train model
            print(f"\nTraining transformer model...")
            training_history = trainer.train(train_loader, test_loader, epochs=50, patience=10)
            
            # Load best model for evaluation
            model.load_state_dict(torch.load('best_transformer_model.pth'))
            
            # Final evaluation
            test_loss, test_acc, predictions, targets = trainer.evaluate(test_loader)
            
            # Calculate medical metrics
            metrics = calculate_medical_metrics(targets, predictions)
            
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS - {dataset_name.upper()} DATASET")
            print(f"{'='*60}")
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
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"transformer_results_{dataset_name}_{timestamp}.pkl"
            
            results = {
                'dataset_name': dataset_name,
                'model_architecture': str(model),
                'training_history': training_history,
                'final_metrics': metrics,
                'label_encoder': label_encoder,
                'input_features': list(X.columns),
                'timestamp': timestamp
            }
            
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"\nResults saved to: {results_file}")
        
        print(f"\n{'='*60}")
        print("TRANSFORMER MODEL TRAINING COMPLETED")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
