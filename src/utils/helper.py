import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import yaml
import xgboost as xgb
import lightgbm as lgb
from src.logger.logs import setup_logger

logger = setup_logger()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved data to {path}")

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved model to {path}")

def load_data(path):
    df = pd.read_csv(path)
    logger.info(f"Loaded data from {path}")
    return df

def load_model(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded model from {path}")
    return obj

def plot_results(y_test, y_pred, r2_score, model_name, model, config):
    logger.info("Generating visualizations...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(config['paths']['performance_plot']), exist_ok=True)
    
    # Performance and residual plots
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k', linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Recycling Rate (%)')
    plt.ylabel('Predicted Recycling Rate (%)')
    plt.title(f'{model_name} Performance\nR² = {r2_score:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    try:
        plt.savefig(config['paths']['performance_plot'], dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance plot to {config['paths']['performance_plot']}")
    except Exception as e:
        logger.error(f"Failed to save performance plot: {str(e)}")
    plt.close()
    
    # Feature importance plot
    os.makedirs(os.path.dirname(config['paths']['feature_importance_plot']), exist_ok=True)
    if model_name in ['LightGBM', 'XGBoost', 'Deep XGBoost']:
        plt.figure(figsize=(12, 8))
        if model_name == 'LightGBM':
            lgb.plot_importance(model, max_num_features=20, importance_type='gain')
        elif model_name in ['XGBoost', 'Deep XGBoost']:
            xgb.plot_importance(model, max_num_features=20, importance_type='gain')
        plt.title('Feature Importance')
        plt.tight_layout()
        try:
            plt.savefig(config['paths']['feature_importance_plot'], dpi=300)
            logger.info(f"Saved feature importance plot to {config['paths']['feature_importance_plot']}")
        except Exception as e:
            logger.error(f"Failed to save feature importance plot: {str(e)}")
        plt.close()
    elif model_name == 'Stacking Ensemble':
        logger.info("Feature importance not plotted for Stacking Ensemble")
    elif model_name in ['Random Forest', 'ExtraTrees']:
        plt.figure(figsize=(12, 8))
        feature_importance = model.feature_importances_
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Feature {i}' for i in range(len(feature_importance))]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{model_name} Feature Importance')
        plt.tight_layout()
        try:
            plt.savefig(config['paths']['feature_importance_plot'], dpi=300)
            logger.info(f"Saved feature importance plot to {config['paths']['feature_importance_plot']}")
        except Exception as e:
            logger.error(f"Failed to save feature importance plot: {str(e)}")
        plt.close()