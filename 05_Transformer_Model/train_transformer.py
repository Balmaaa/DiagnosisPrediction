#!/usr/bin/env python3
"""
DEDICATED TRANSFORMER TRAINING SCRIPT
Ensures the main() function is actually executed
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("STARTING TRANSFORMER TRAINING EXECUTION")
print("="*80)
print("This script ensures the main() function is called")
print("="*80)

# Import and execute main function
from transformer_model import main

if __name__ == "__main__":
    print("\n[OK] Training script loaded successfully")
    print("[OK] About to call main() function...")
    print("="*80)
    
    try:
        main()
        print("\n" + "="*80)
        print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\n[ERROR] TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
