from backend_predictor import AppendicitisPredictor

# Test the real-time metrics function
predictor = AppendicitisPredictor()

print('=== TESTING REAL-TIME METRICS ===')
metrics = predictor.get_realtime_metrics()

if 'error' in metrics:
    print(f'ERROR: {metrics["error"]}')
else:
    for model_name, metrics_data in metrics.items():
        if 'error' in metrics_data:
            print(f'{model_name}: ERROR - {metrics_data["error"]}')
        else:
            print(f'{model_name}:')
            print(f'  Precision: {metrics_data["precision"]:.4f}')
            print(f'  Sensitivity: {metrics_data["sensitivity"]:.4f}')
            print(f'  Specificity: {metrics_data["specificity"]:.4f}')
            print()
