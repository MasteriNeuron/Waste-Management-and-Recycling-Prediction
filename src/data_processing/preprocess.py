import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import yaml
import os
from src.logger.logs import setup_logger

logger = setup_logger()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def create_hybrid_features(df):
    logger.info("Creating hybrid features...")
    df = df.copy()
    
    # Parse coordinates
    coords = df['Landfill Location (Lat, Long)'].str.split(',', expand=True)
    df['Landfill_Lat'] = coords[0].astype(float)
    df['Landfill_Long'] = coords[1].astype(float)
    
    # Key ratios and interactions
    df['Waste_Per_Capita'] = df['Waste Generated (Tons/Day)'] / (df['Population Density (People/km²)'] + 1)
    df['Cost_Efficiency'] = df['Cost of Waste Management (₹/Ton)'] / (df['Municipal Efficiency Score (1-10)'] + 1)
    df['Efficiency_x_Campaigns'] = df['Municipal Efficiency Score (1-10)'] * df['Awareness Campaigns Count']
    df['Population_x_Waste'] = df['Population Density (People/km²)'] * df['Waste Generated (Tons/Day)']
    
    # Advanced target encoding with smoothing
    for col in ['City/District', 'Waste Type', 'Disposal Method']:
        target_mean = df.groupby(col)['Recycling Rate (%)'].transform('mean')
        target_std = df.groupby(col)['Recycling Rate (%)'].transform('std').fillna(0)
        df[f'{col}_TargetMean'] = target_mean
        df[f'{col}_TargetStd'] = target_std
        
        global_mean = df['Recycling Rate (%)'].mean()
        counts = df.groupby(col).size()
        df[f'{col}_Count'] = df[col].map(counts)
        smoothing = 10
        df[f'{col}_SmoothedTarget'] = (
            (df[f'{col}_TargetMean'] * df[f'{col}_Count'] + global_mean * smoothing) / 
            (df[f'{col}_Count'] + smoothing)
        )
    
    # City and waste type statistics
    city_stats = df.groupby('City/District').agg({
        'Recycling Rate (%)': ['mean', 'std'],
        'Municipal Efficiency Score (1-10)': 'mean'
    }).reset_index()
    city_stats.columns = ['City/District', 'City_Recycling_Mean', 'City_Recycling_Std', 'City_Efficiency_Mean']
    df = df.merge(city_stats, on='City/District', how='left')
    
    waste_stats = df.groupby('Waste Type').agg({
        'Recycling Rate (%)': ['mean', 'std']
    }).reset_index()
    waste_stats.columns = ['Waste Type', 'WasteType_Recycling_Mean', 'WasteType_Recycling_Std']
    df = df.merge(waste_stats, on='Waste Type', how='left')
    
    # Log transformations
    for col in ['Waste Generated (Tons/Day)', 'Population Density (People/km²)', 'Cost of Waste Management (₹/Ton)']:
        df[f'Log_{col}'] = np.log1p(df[col])
    
    # Polynomial features
    df['Efficiency_Squared'] = df['Municipal Efficiency Score (1-10)'] ** 2
    df['Campaigns_Squared'] = df['Awareness Campaigns Count'] ** 2
    
    # Interaction with target-encoded features
    df['City_Waste_Interaction'] = df['City/District_TargetMean'] * df['Waste Generated (Tons/Day)']
    df['WasteType_Efficiency_Interaction'] = df['Waste Type_TargetMean'] * df['Municipal Efficiency Score (1-10)']
    
    logger.info("Feature engineering completed")
    logger.info(f"Final DataFrame shape: {df.shape}, columns: {list(df.columns)}")
    return df

def hybrid_augmentation(df, target_col='Recycling Rate (%)'):
    logger.info("Augmenting data...")
    augmented_dfs = [df]
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df['Target_Bin'] = pd.qcut(df[target_col], q=10, labels=False, duplicates='drop')
    
    for bin_val in df['Target_Bin'].unique():
        bin_data = df[df['Target_Bin'] == bin_val]
        if len(bin_data) > 5:
            n_synthetic = min(100, len(bin_data))
            synthetic_samples = []
            for _ in range(n_synthetic):
                idx1, idx2 = np.random.choice(bin_data.index, 2, replace=True)
                alpha = np.random.beta(2, 2)
                synthetic = {}
                for col in numeric_cols:
                    if col != 'Target_Bin':
                        synthetic[col] = alpha * df.loc[idx1, col] + (1-alpha) * df.loc[idx2, col]
                synthetic_samples.append(synthetic)
            augmented_dfs.append(pd.DataFrame(synthetic_samples))
    
    mixup_df = df.copy()
    for i in range(len(mixup_df)):
        if np.random.random() < 0.5:
            j = np.random.randint(0, len(mixup_df))
            lambda_param = np.random.beta(0.2, 0.2)
            for col in numeric_cols:
                if col in mixup_df.columns and col != 'Target_Bin':
                    mixup_df.loc[i, col] = lambda_param * mixup_df.loc[i, col] + (1 - lambda_param) * mixup_df.loc[j, col]
    augmented_dfs.append(mixup_df)
    
    noise_df = df.copy()
    for col in numeric_cols:
        if target_col not in col and 'Target' not in col:
            noise = np.random.normal(0, 0.005 * noise_df[col].std(), len(noise_df))
            noise_df[col] = noise_df[col] + noise
    augmented_dfs.append(noise_df)
    
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    final_df = final_df.drop('Target_Bin', axis=1, errors='ignore')
    logger.info(f"Augmented dataset size: {len(final_df)} (from {len(df)})")
    #logger.info(f"final columns in argumented dataset {final_df.columns}")
    return final_df

def preprocess_data(config):
    logger.info("Starting data preprocessing...")
    df = pd.read_csv(config['paths']['raw_data'])
    logger.info(f"Original dataset shape: {df.shape}")
    
    df_featured = create_hybrid_features(df)
    df_augmented = hybrid_augmentation(df_featured)
    os.makedirs(os.path.dirname(config['paths']['processed_data']), exist_ok=True)
    df_augmented.to_csv(config['paths']['processed_data'], index=False)
    
    feature_cols = [col for col in df_augmented.columns if col not in [
        'Recycling Rate (%)', 'City/District', 'Waste Type', 'Disposal Method',
        'Landfill Name', 'Landfill Location (Lat, Long)'
    ]]
    
    X = df_augmented[feature_cols].fillna(df_augmented[feature_cols].mean())
    y = df_augmented['Recycling Rate (%)']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    selector = SelectKBest(score_func=mutual_info_regression, k=min(config['parameters']['n_features'], len(feature_cols)))
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=config['parameters']['test_size'], random_state=config['parameters']['random_state']
    )
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler, selector, selected_features