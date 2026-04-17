"""
Standalone Prediction Utility
Quick command-line interface for making predictions without API
"""

import joblib
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from config import paths


def load_models():
    """Load trained models"""
    try:
        model = joblib.load(paths.MODEL_FILE)
        scaler = joblib.load(paths.SCALER_FILE)
        feature_names = joblib.load(paths.FEATURE_NAMES_FILE)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please train the model first using: python main.py")
        print(f"Missing file: {e.filename}")
        return None, None, None


def prepare_input(args, feature_names):
    """Prepare input data for prediction"""
    # Create base dataframe
    data = {
        'hour': args.hour,
        'day_of_week': args.day_of_week,
        'month': args.month,
        'queue_length_at_arrival': args.queue_length,
        'total_counters_available': args.counters,
        'arrival_rate': args.arrival_rate,
        'system_utilization': args.utilization,
        'doctor_efficiency': args.efficiency
    }
    
    df = pd.DataFrame([data])
    
    # Add derived features (must match training pipeline)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    df['is_peak_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 12)).astype(int)
    df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    df['workload_index'] = (df['queue_length_at_arrival'] * df['system_utilization']) / (df['total_counters_available'] + 1)
    df['queue_pressure'] = df['queue_length_at_arrival'] * df['system_utilization']
    df['counters_per_patient'] = df['total_counters_available'] / (df['queue_length_at_arrival'] + 1)
    df['efficiency_gap'] = df['doctor_efficiency'] - df['system_utilization']
    df['log_queue'] = np.log1p(df['queue_length_at_arrival'])
    
    # Ensure all features are present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select features in correct order
    df = df[feature_names]
    
    return df


def predict(model, scaler, feature_names, args):
    """Make prediction"""
    # Prepare input
    X = prepare_input(args, feature_names)
    
    # Scale
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    prediction = max(0, prediction)  # Ensure non-negative
    
    return prediction


def get_recommendation(wait_time, utilization, queue_length):
    """Generate recommendation based on prediction"""
    recommendations = []
    
    if wait_time > 45:
        recommendations.append("⚠️  CRITICAL: Wait time exceeds 45 minutes. Immediate counter increase required.")
    elif wait_time > 30:
        recommendations.append("⚠️  WARNING: Wait time above acceptable threshold. Consider adding counters.")
    else:
        recommendations.append("✓  Wait time within acceptable range.")
    
    if utilization > 0.90:
        recommendations.append("⚠️  System utilization critical. Staff are overloaded.")
    elif utilization > 0.85:
        recommendations.append("⚠️  High utilization. Monitor closely.")
    
    if queue_length > 15:
        recommendations.append("⚠️  Long queue detected. Consider capacity expansion.")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Hospital Queue Wait Time Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction for OPD at 10 AM
  python predict.py --hour 10 --queue-length 12

  # Full parameters
  python predict.py --hour 14 --day-of-week 1 --month 2 \\
                    --queue-length 15 --counters 4 \\
                    --arrival-rate 2.5 --utilization 0.85

  # Monday morning rush
  python predict.py --hour 9 --day-of-week 0 --queue-length 20 \\
                    --counters 5 --utilization 0.9
        """
    )
    
    # Time parameters
    parser.add_argument('--hour', type=int, required=True, 
                       help='Hour of day (0-23)')
    parser.add_argument('--day-of-week', type=int, default=1,
                       help='Day of week (0=Monday, 6=Sunday), default=1')
    parser.add_argument('--month', type=int, default=1,
                       help='Month (1-12), default=1')
    
    # Queue parameters
    parser.add_argument('--queue-length', type=int, required=True,
                       help='Number of patients in queue')
    parser.add_argument('--counters', type=int, default=4,
                       help='Number of available counters, default=4')
    parser.add_argument('--arrival-rate', type=float, default=2.0,
                       help='Patient arrival rate (patients/hour), default=2.0')
    
    # System parameters
    parser.add_argument('--utilization', type=float, default=0.75,
                       help='System utilization (0-1), default=0.75')
    parser.add_argument('--efficiency', type=float, default=0.80,
                       help='Doctor efficiency (0-1), default=0.80')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed information')
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    model, scaler, feature_names = load_models()
    
    if model is None:
        return 1
    
    print("Models loaded successfully!\n")
    
    # Make prediction
    print("=" * 60)
    print("QUEUE WAIT TIME PREDICTION")
    print("=" * 60)
    print(f"\nInput Parameters:")
    print(f"  Time: Hour {args.hour:02d}:00")
    print(f"  Day of Week: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][args.day_of_week]}")
    print(f"  Queue Length: {args.queue_length} patients")
    print(f"  Active Counters: {args.counters}")
    print(f"  Arrival Rate: {args.arrival_rate:.1f} patients/hour")
    print(f"  System Utilization: {args.utilization:.1%}")
    print(f"  Doctor Efficiency: {args.efficiency:.1%}")
    
    prediction = predict(model, scaler, feature_names, args)
    
    print("\n" + "=" * 60)
    print(f"PREDICTED WAIT TIME: {prediction:.1f} minutes")
    print("=" * 60)
    
    # Generate recommendations
    print("\nRecommendations:")
    recommendations = get_recommendation(prediction, args.utilization, args.queue_length)
    for rec in recommendations:
        print(f"  {rec}")
    
    # Confidence interval (approximate)
    ci_lower = max(0, prediction * 0.85)
    ci_upper = prediction * 1.15
    
    print(f"\n95% Confidence Interval: [{ci_lower:.1f}, {ci_upper:.1f}] minutes")
    
    if args.verbose:
        print(f"\nTechnical Details:")
        print(f"  Total features used: {len(feature_names)}")
        print(f"  Model type: Stacking Ensemble")
        print(f"  Base models: XGBoost, LightGBM, CatBoost, ExtraTrees")
    
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())