# src/pipelines/prediction_pipelines.py
import pandas as pd
import numpy as np
import pickle
import os
import yaml
from src.logger.logs import setup_logger
from src.utils.helper import load_model as helper_load_model

logger = setup_logger()

def load_config(config_path='src/config/config.yaml'):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def load_artifacts(config):
    """Load all required artifacts for prediction"""
    artifacts = {}
    
    try:
        # Load model
        artifacts['model'] = helper_load_model(config['paths']['model'])
        
        # Load scaler
        artifacts['scaler'] = helper_load_model(config['paths']['scaler'])
        
        # Load feature selector
        artifacts['selector'] = helper_load_model(config['paths']['selector'])
        
        # Load model name
        artifacts['model_name'] = helper_load_model(config['paths']['model_name'])
        
        # Load target encodings
        artifacts['target_encodings'] = helper_load_model(config['paths']['target_encodings'])
        
        logger.info("All artifacts loaded successfully")
        return artifacts
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {str(e)}")
        raise

def apply_feature_engineering(df, target_encodings):
    """Apply the same feature engineering that was done during training"""
    logger.info("Applying feature engineering for prediction...")
    df_processed = df.copy()
    
    # Parse coordinates
    coords = df_processed['Landfill Location (Lat, Long)'].str.split(',', expand=True)
    df_processed['Landfill_Lat'] = coords[0].astype(float)
    df_processed['Landfill_Long'] = coords[1].astype(float)
    
    # Key ratios and interactions
    df_processed['Waste_Per_Capita'] = df_processed['Waste Generated (Tons/Day)'] / (df_processed['Population Density (People/km²)'] + 1)
    df_processed['Cost_Efficiency'] = df_processed['Cost of Waste Management (₹/Ton)'] / (df_processed['Municipal Efficiency Score (1-10)'] + 1)
    df_processed['Efficiency_x_Campaigns'] = df_processed['Municipal Efficiency Score (1-10)'] * df_processed['Awareness Campaigns Count']
    df_processed['Population_x_Waste'] = df_processed['Population Density (People/km²)'] * df_processed['Waste Generated (Tons/Day)']
    
    # Apply target encodings from training
    categorical_cols = ['City/District', 'Waste Type', 'Disposal Method']
    
    for col in categorical_cols:
        mean_map = target_encodings.get(f'{col}_TargetMean', {})
        std_map = target_encodings.get(f'{col}_TargetStd', {})
        count_map = target_encodings.get(f'{col}_Count', {})
        smoothed_map = target_encodings.get(f'{col}_SmoothedTarget', {})
        
        global_mean = df_processed['Waste Generated (Tons/Day)'].mean() if 'Waste Generated (Tons/Day)' in df_processed else 0
        
        df_processed[f'{col}_TargetMean'] = df_processed[col].map(mean_map).fillna(np.mean(list(mean_map.values())) if mean_map else global_mean)
        df_processed[f'{col}_TargetStd'] = df_processed[col].map(std_map).fillna(np.mean(list(std_map.values())) if std_map else 0)
        df_processed[f'{col}_Count'] = df_processed[col].map(count_map).fillna(np.mean(list(count_map.values())) if count_map else 1)
        df_processed[f'{col}_SmoothedTarget'] = df_processed[col].map(smoothed_map).fillna(np.mean(list(smoothed_map.values())) if smoothed_map else global_mean)
    
    # Apply city and waste type statistics
    city_stats = pd.DataFrame.from_dict(target_encodings.get('City_Stats', {}), orient='index').reset_index()
    city_stats.columns = ['City/District', 'City_Recycling_Mean', 'City_Recycling_Std', 'City_Efficiency_Mean']
    df_processed = df_processed.merge(city_stats, on='City/District', how='left')
    
    waste_stats = pd.DataFrame.from_dict(target_encodings.get('Waste_Stats', {}), orient='index').reset_index()
    waste_stats.columns = ['Waste Type', 'WasteType_Recycling_Mean', 'WasteType_Recycling_Std']
    df_processed = df_processed.merge(waste_stats, on='Waste Type', how='left')
    
    # Log transformations
    for col in ['Waste Generated (Tons/Day)', 'Population Density (People/km²)', 'Cost of Waste Management (₹/Ton)']:
        if col in df_processed.columns:
            df_processed[f'Log_{col}'] = np.log1p(df_processed[col])
    
    # Polynomial features
    df_processed['Efficiency_Squared'] = df_processed['Municipal Efficiency Score (1-10)'] ** 2
    df_processed['Campaigns_Squared'] = df_processed['Awareness Campaigns Count'] ** 2
    
    # Interaction with target-encoded features
    if 'City/District_TargetMean' in df_processed.columns and 'Waste Generated (Tons/Day)' in df_processed.columns:
        df_processed['City_Waste_Interaction'] = df_processed['City/District_TargetMean'] * df_processed['Waste Generated (Tons/Day)']
    if 'Waste Type_TargetMean' in df_processed.columns and 'Municipal Efficiency Score (1-10)' in df_processed.columns:
        df_processed['WasteType_Efficiency_Interaction'] = df_processed['Waste Type_TargetMean'] * df_processed['Municipal Efficiency Score (1-10)']
    
    # Ensure Landfill Capacity and Year are present (they might come from input)
    if 'Landfill Capacity (Tons)' not in df_processed.columns:
        df_processed['Landfill Capacity (Tons)'] = 10000
    
    if 'Year' not in df_processed.columns:
        df_processed['Year'] = 2024
    
    logger.info("Feature engineering completed")
    return df_processed


def preprocess_for_prediction(df, artifacts):
    """Preprocess new data using the same feature engineering as training"""
    logger.info("Preprocessing data for prediction...")
    
    # Apply feature engineering using stored target encodings
    df_processed = apply_feature_engineering(df, artifacts['target_encodings'])
    
    # EXACT feature order from training (as you provided)
    expected_features = [
        'Waste Generated (Tons/Day)', 'Population Density (People/km²)', 
        'Municipal Efficiency Score (1-10)', 'Cost of Waste Management (₹/Ton)', 
        'Awareness Campaigns Count', 'Landfill Capacity (Tons)', 'Year', 
        'Landfill_Lat', 'Landfill_Long', 'Waste_Per_Capita', 'Cost_Efficiency', 
        'Efficiency_x_Campaigns', 'Population_x_Waste', 'City/District_TargetMean', 
        'City/District_TargetStd', 'City/District_Count', 'City/District_SmoothedTarget', 
        'Waste Type_TargetMean', 'Waste Type_TargetStd', 'Waste Type_Count', 
        'Waste Type_SmoothedTarget', 'Disposal Method_TargetMean', 
        'Disposal Method_TargetStd', 'Disposal Method_Count', 
        'Disposal Method_SmoothedTarget', 'City_Recycling_Mean', 
        'City_Recycling_Std', 'City_Efficiency_Mean', 'WasteType_Recycling_Mean', 
        'WasteType_Recycling_Std', 'Log_Waste Generated (Tons/Day)', 
        'Log_Population Density (People/km²)', 'Log_Cost of Waste Management (₹/Ton)', 
        'Efficiency_Squared', 'Campaigns_Squared', 'City_Waste_Interaction', 
        'WasteType_Efficiency_Interaction'
    ]
    
    # Ensure all expected features are present with default values
    missing_features = set(expected_features) - set(df_processed.columns)
    if missing_features:
        logger.warning(f"Missing features in input data: {missing_features}")
        for feature in missing_features:
            if feature == 'Landfill Capacity (Tons)':
                df_processed[feature] = 10000  # Reasonable default
            elif feature == 'Year':
                df_processed[feature] = 2024  # Current year
            elif feature in ['City_Recycling_Mean', 'City_Recycling_Std', 'City_Efficiency_Mean',
                           'WasteType_Recycling_Mean', 'WasteType_Recycling_Std']:
                # These are statistical features, use 0 as default
                df_processed[feature] = 0
            elif 'Target' in feature or 'Smoothed' in feature:
                # Target encoding features, use 0 as default
                df_processed[feature] = 0
            else:
                df_processed[feature] = 0  # Default for other features
    
    # Select only the required columns in the EXACT same order as training
    X = df_processed[expected_features].fillna(0)
    
    logger.info(f"Preprocessed data shape: {X.shape}")
    logger.info(f"Features used (in order): {list(X.columns)}")
    return X, expected_features

def predict(model_data, X, model_name, scaler, selector):
    """Make predictions using the loaded model"""
    logger.info(f"Making predictions with model: {model_name}")
    
    try:
        # Scale the data
        X_scaled = scaler.transform(X)
        
        if model_name == "Deep XGBoost":
            # Use all features (no selection)
            predictions = model_data.predict(X_scaled)
        
        elif model_name == "Stacking Ensemble":
            # Feature selection for base models
            X_selected = selector.transform(X_scaled)
            
            # Generate meta-features from base models
            meta_features = np.column_stack([
                model_data['lgb_model'].predict(X_selected),
                model_data['xgb_model'].predict(X_selected),
                model_data['rf_model'].predict(X_selected),
                model_data['gb_model'].predict(X_selected),
                model_data['et_model'].predict(X_selected)
            ])
            
            # Final prediction from meta-model
            predictions = model_data['meta_model'].predict(meta_features)
        
        else:
            # Feature selection for individual models
            X_selected = selector.transform(X_scaled)
            predictions = model_data.predict(X_selected)
        
        logger.info(f"Predictions completed. Shape: {predictions.shape}")
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

def predict_single_instance(instance_data, config, artifacts):
    """Predict for a single data instance (dictionary format)"""
    logger.info("Predicting for single instance...")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([instance_data])
        
        # Preprocess the data
        X_processed, feature_cols = preprocess_for_prediction(df, artifacts)
        
        # Make prediction
        prediction = predict(
            artifacts['model'], 
            X_processed, 
            artifacts['model_name'], 
            artifacts['scaler'], 
            artifacts['selector']
        )
        
        logger.info(f"Single instance prediction completed: {prediction[0]:.2f}%")
        return prediction[0]
        
    except Exception as e:
        logger.error(f"Single instance prediction failed: {str(e)}")
        raise

def predict_batch(input_data, config, artifacts):
    """Main prediction function for batch processing"""
    logger.info("Starting batch prediction...")
    
    try:
        # Handle different input types
        if isinstance(input_data, str):
            # Input is a file path
            df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            # Input is already a DataFrame
            df = input_data.copy()
        else:
            raise ValueError("Input data must be a file path or pandas DataFrame")
        
        # Preprocess the data
        X_processed, feature_cols = preprocess_for_prediction(df, artifacts)
        
        # Make predictions
        predictions = predict(
            artifacts['model'], 
            X_processed, 
            artifacts['model_name'], 
            artifacts['scaler'], 
            artifacts['selector']
        )
        
        # Add predictions to original dataframe
        df['Predicted_Recycling_Rate'] = predictions
        
        logger.info(f"Batch prediction completed. Generated {len(predictions)} predictions")
        return df
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise

def save_predictions(predictions_df, output_path):
    """Save predictions to file"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {str(e)}")
        raise

def initialize_prediction_system(config_path='src/config/config.yaml'):
    """Initialize the prediction system by loading all artifacts"""
    try:
        config = load_config(config_path)
        artifacts = load_artifacts(config)
        logger.info("Prediction system initialized successfully")
        return config, artifacts
    except Exception as e:
        logger.error(f"Failed to initialize prediction system: {str(e)}")
        return None, None

def example_usage():
    """Example of how to use the prediction pipeline"""
    try:
        # Initialize
        config, artifacts = initialize_prediction_system()
        
        if config and artifacts:
            # Example 1: Single instance prediction
            sample_data = {
                'City/District': 'Mumbai',
                'Waste Type': 'Plastic',
                'Disposal Method': 'Recycling',
                'Waste Generated (Tons/Day)': 150.5,
                'Population Density (People/km²)': 25000,
                'Cost of Waste Management (₹/Ton)': 1200,
                'Municipal Efficiency Score (1-10)': 7,
                'Awareness Campaigns Count': 5,
                'Landfill Name': 'Mumbai Landfill',
                'Landfill Location (Lat, Long)': '19.0760,72.8777'
            }
            
            prediction = predict_single_instance(sample_data, config, artifacts)
            print(f"Predicted Recycling Rate: {prediction:.2f}%")
            
    except Exception as e:
        logger.error(f"Example usage failed: {e}")

if __name__ == '__main__':
    example_usage()