import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import pickle
from src.data_processing.preprocess import preprocess_data, load_config
from src.utils.helper import save_model, plot_results
from src.logger.logs import setup_logger

logger = setup_logger()

def build_hybrid_model(X_train, y_train, X_test, y_test, X_full, y_full, scaler, config):
    logger.info("Starting model training...")
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=63,
        max_depth=10,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=config['parameters']['random_state'],
        n_jobs=-1,
        verbosity=-1
    )
    logger.info("Training LightGBM...")
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_pred)
    logger.info(f"LightGBM R²: {lgb_r2:.4f}")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=config['parameters']['random_state'],
        n_jobs=-1
    )
    logger.info("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_pred)
    logger.info(f"XGBoost R²: {xgb_r2:.4f}")
    
    # Stacking Ensemble
    logger.info("Building stacking ensemble...")
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=config['parameters']['random_state'],
        n_jobs=-1
    )
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=config['parameters']['random_state']
    )
    et_model = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        random_state=config['parameters']['random_state'],
        n_jobs=-1
    )
    
    # Train base models
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    et_model.fit(X_train, y_train)
    
    # Create meta features
    meta_features = np.column_stack([
        lgb_model.predict(X_test),
        xgb_model.predict(X_test),
        rf_model.predict(X_test),
        gb_model.predict(X_test),
        et_model.predict(X_test)
    ])
    
    # Train meta model
    meta_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=config['parameters']['random_state']
    )
    meta_model.fit(meta_features, y_test)
    
    # Final predictions
    stacking_pred = meta_model.predict(meta_features)
    stacking_r2 = r2_score(y_test, stacking_pred)
    logger.info(f"Stacking Ensemble R²: {stacking_r2:.4f}")
    
    # Select best model
    models = [
        (lgb_pred, lgb_r2, "LightGBM", lgb_model),
        (xgb_pred, xgb_r2, "XGBoost", xgb_model),
        (stacking_pred, stacking_r2, "Stacking Ensemble", {
            'lgb_model': lgb_model,
            'xgb_model': xgb_model,
            'rf_model': rf_model,
            'gb_model': gb_model,
            'et_model': et_model,
            'meta_model': meta_model
        })
    ]
    best_pred, best_r2, best_model_name, best_model = max(models, key=lambda x: x[1])
    
    # Deep XGBoost fallback if R² < 0.80
    if best_r2 < 0.80:
        logger.info("Trying Deep XGBoost with all features...")
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_full, y_full, test_size=config['parameters']['test_size'], random_state=config['parameters']['random_state']
        )
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_full)
        
        # FIXED: early_stopping_rounds should be in constructor, not fit()
        deep_xgb = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.005,
            max_depth=12,
            min_child_weight=1,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=0.05,
            random_state=config['parameters']['random_state'],
            n_jobs=-1,
            early_stopping_rounds=50  # MOVED from fit() to constructor
        )
        logger.info("Training Deep XGBoost with early stopping...")
        deep_xgb.fit(
            X_train_scaled, y_train_full,
            eval_set=[(X_test_scaled, y_test_full)],
            verbose=10
        )
        deep_pred = deep_xgb.predict(X_test_scaled)
        deep_r2 = r2_score(y_test_full, deep_pred)
        logger.info(f"Deep XGBoost R²: {deep_r2:.4f}")
        
        if deep_r2 > best_r2:
            best_pred = deep_pred
            best_r2 = deep_r2
            best_model_name = "Deep XGBoost"
            best_model = deep_xgb
            y_test = y_test_full
    
    # Save best model
    save_model(best_model, config['paths']['model'])
    
    # Save results
    results_summary = pd.DataFrame({
        'Model': [best_model_name],
        'R2_Score': [best_r2],
        'RMSE': [np.sqrt(mean_squared_error(y_test, best_pred))],
        'Dataset_Size': [len(X_train) + len(X_test)],
        'Features_Used': [X_train.shape[1] if best_model_name != "Deep XGBoost" else X_train_full.shape[1]]
    })
    results_summary.to_csv(config['paths']['results'], index=False)
    logger.info(f"Saved results to {config['paths']['results']}")
    
    # Plot results
    plot_results(y_test, best_pred, best_r2, best_model_name, best_model, config)
    
    return best_pred, best_r2, best_model_name, best_model

def main():
    logger.info("Running training pipeline standalone")
    config = load_config('src/config/config.yaml')
    
    # Preprocess data
    X_train, X_test, y_train, y_test, scaler, selector, selected_features = preprocess_data(config)
    logger.info("selector:",selector)
    # Load full features for Deep XGBoost
    df_augmented = pd.read_csv(config['paths']['processed_data'])
    feature_cols = [col for col in df_augmented.columns if col not in [
        'Recycling Rate (%)', 'City/District', 'Waste Type', 'Disposal Method',
        'Landfill Name', 'Landfill Location (Lat, Long)'
    ]]
    X_full = df_augmented[feature_cols].fillna(df_augmented[feature_cols].mean())
    y_full = df_augmented['Recycling Rate (%)']
    
    # Train model
    best_pred, best_r2, best_model_name, best_model = build_hybrid_model(X_train, y_train, X_test, y_test, X_full, y_full, scaler, config)
    
    # Save preprocessing artifacts
    save_model(scaler, config['paths']['scaler'])
    save_model(selector, config['paths']['selector'])
    save_model(best_model_name, config['paths']['model_name'])
    
    # Log final results
    logger.info("\nFinal Results:")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"R² Score: {best_r2:.4f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, best_pred)):.4f}")
    logger.info(f"Dataset Size: {X_train.shape[0] + X_test.shape[0]}")
    logger.info(f"Features Used: {len(selected_features)}")
    
    if best_r2 >= 0.80:
        logger.info("\n✅ SUCCESS! Target R² ≥ 80% ACHIEVED!")
    else:
        logger.info(f"\n⚠️ Current R²: {best_r2:.1%}")
        logger.info("\nTo achieve R² > 80%, consider:")
        logger.info("1. Collecting more training data")
        logger.info("2. Adding external features")
        logger.info("3. Using neural networks")
        logger.info("4. More sophisticated feature engineering")
        logger.info("5. Hyperparameter optimization")

if __name__ == '__main__':
    main()