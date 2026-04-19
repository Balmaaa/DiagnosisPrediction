"""Test Transformer import and data loading"""
import sys
import os

# Add paths
sys.path.insert(0, '09_GUI_Application')
sys.path.insert(0, '06_Decision_Trees')
sys.path.insert(0, '07_Gradient_Boosting')
sys.path.insert(0, '08_XGBoost')
sys.path.insert(0, '05_Transformer_Model')

print("="*60)
print("TESTING TRANSFORMER IMPORTS")
print("="*60)

try:
    print("\n[1/4] Testing DecisionTreeModel import...")
    from decision_tree_model import DecisionTreeModel
    dt = DecisionTreeModel()
    X_train, X_test, y_train, y_test = dt.load_unified_data()
    print(f"   DecisionTreeModel: OK, data shape: {X_train.shape}")
except Exception as e:
    print(f"   DecisionTreeModel: FAILED - {e}")
    sys.exit(1)

try:
    print("\n[2/4] Testing GradientBoostingModel import...")
    from gradient_boosting_model import GradientBoostingModel
    gb = GradientBoostingModel()
    X_train, X_test, y_train, y_test = gb.load_unified_data()
    print(f"   GradientBoostingModel: OK, data shape: {X_train.shape}")
except Exception as e:
    print(f"   GradientBoostingModel: FAILED - {e}")
    sys.exit(1)

try:
    print("\n[3/4] Testing XGBoostModel import...")
    from xgboost_model import XGBoostModel
    xgb = XGBoostModel()
    X_train, X_test, y_train, y_test = xgb.load_unified_data()
    print(f"   XGBoostModel: OK, data shape: {X_train.shape}")
except Exception as e:
    print(f"   XGBoostModel: FAILED - {e}")
    sys.exit(1)

try:
    print("\n[4/4] Testing Transformer import...")
    from transformer_model import AdvancedTabularTransformer
    print(f"   AdvancedTabularTransformer: OK")
except Exception as e:
    print(f"   AdvancedTabularTransformer: FAILED - {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL IMPORTS SUCCESSFUL!")
print("="*60)

# Test backend predictor
try:
    print("\nTesting backend predictor...")
    from backend_predictor import AppendicitisPredictor
    predictor = AppendicitisPredictor()
    print(f"Models loaded: {list(predictor.models.keys())}")
except Exception as e:
    print(f"Backend predictor failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SYSTEM READY!")
print("="*60)
