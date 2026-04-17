"""
Main Training Pipeline
=======================
AI-Based Queue-Time Prediction and Smart Counter Allocation Decision Tool

Complete end-to-end training pipeline integrating all modules:
1. Data Collection and Preprocessing (Module 1)
2. Predictive Modeling (Module 2)
3. Counter Allocation (Module 3)
4. Performance Monitoring (Module 5)

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path
import json

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / 'modules'))

from config import paths, system_config
from modules.module1_data_preprocessing import load_and_preprocess_data
from modules.module2_predictive_modelling import train_and_compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(paths.TRAINING_LOG)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline execution"""
    
    try:
        logger.info("="*80)
        logger.info(f"{system_config.project_name} - Training Pipeline")
        logger.info(f"Version {system_config.version}")
        logger.info("="*80)
        
        # ====================================================================
        # PHASE 1-2: Data Collection and Preprocessing (Module 1)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 1-2: Data Collection and Preprocessing")
        logger.info("="*80)
        
        X, y, preprocessor = load_and_preprocess_data(paths.RAW_DATA)
        
        logger.info(f"\n[SUMMARY] Preprocessing Complete")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Samples: {X.shape[0]:,}")
        logger.info(f"  Target range: [{y.min():.1f}, {y.max():.1f}] minutes")
        
        # ====================================================================
        # PHASE 3: Model Development and Training (Module 2)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: Model Development and Training")
        logger.info("="*80)
        
        best_model_name, metrics = train_and_compare_models(X, y)

        
        logger.info(f"\n[SUMMARY] Model Training Complete")
        logger.info(f"  Best Model: {best_model_name}")
        logger.info(f"  Test RMSE: {metrics['rmse']:.2f} minutes")
        logger.info(f"  Test MAE: {metrics['mae']:.2f} minutes")
        logger.info(f"  Test R²: {metrics['r2']:.4f}")
        
        # ====================================================================
        # PHASE 4: Test Counter Allocation (Module 3)
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: Testing Counter Allocation")
        logger.info("="*80)
        
        from modules.module3_counter_allocation import AllocationEngine
        
        allocation_engine = AllocationEngine(method='linear_programming')
        
        # Example predictions for testing
        department_predictions = {
            'OPD': 28.5,
            'Diagnostics': 22.3,
            'Pharmacy': 15.1
        }
        
        current_allocation = {
            'OPD': 4,
            'Diagnostics': 3,
            'Pharmacy': 3
        }
        
        recommendations = allocation_engine.generate_recommendations(
            department_predictions,
            current_allocation
        )
        
        alerts = allocation_engine.generate_alerts(recommendations)
        allocation_engine.save_recommendations()
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("="*80)
        
        summary = {
            'status': 'SUCCESS',
            'data': {
                'samples': int(X.shape[0]),
                'features': int(X.shape[1]),
                'target_range': [float(y.min()), float(y.max())]
            },
            'model': {
                'name': best_model_name,
                'rmse': float(metrics['rmse']),
                'mae': float(metrics['mae']),
                'r2_score': float(metrics['r2']),
            },
            'allocation': {
                'recommendations_generated': len(recommendations),
                'alerts_generated': len(alerts)
            }
        }
        
        # Save summary
        summary_path = paths.OUTPUTS_DIR / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n[FILES SAVED]")
        logger.info(f"  Model: {paths.TRAINED_MODEL}")
        logger.info(f"  Metrics: {paths.COMPARISON_JSON}")
        logger.info(f"  Feature Importance: {paths.FEATURE_IMPORTANCE}")
        logger.info(f"  Allocation Logs: {paths.ALLOCATION_LOGS}")
        logger.info(f"  Training Log: {paths.TRAINING_LOG}")
        logger.info(f"  Summary: {summary_path}")
        
        logger.info(f"\n[NEXT STEPS]")
        logger.info(f"  1. Review model metrics in {paths.COMPARISON_JSON}")
        logger.info(f"  2. Start dashboard: streamlit run dashboard/app.py")
        logger.info(f"  3. Monitor performance using Module 5")
        logger.info(f"  4. Deploy system for hospital testing")
        
        logger.info("\n" + "="*80)
        logger.info("[SUCCESS] All phases completed successfully!")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n[ERROR] Training pipeline failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)