import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df, config):
    """
    データ前処理を行う関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        前処理するデータフレーム
    config : dict
        前処理の設定
    
    Returns:
    --------
    pandas.DataFrame
        前処理後のデータフレーム
    """
    
    processed_df = df.copy()
    
    if 'convert_types' in config and config['convert_types']:
        for col in processed_df.columns:
            try:
                if dtype == 'datetime':
                    processed_df[col] = pd.to_datetime(processed_df[col])
                else:
                    processed_df[col] = processed_df[col].astype(dtype)
            except Exception as e:
                print(f"列 {col} の型変換に失敗しました: {e}")
                
    if 'handle_missing' in config and config['handle_missing']:
        for col, method in config['handle_missing'].items():
            if col in processed_df.columns and processed_df[col].isnull().any():
                if method == 'drop':
                    processed_df = processed_df.dropna(subset=[col])
                elif method == 'mean':
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                elif method == 'median':
                    processed_df[col] = processed_df[col].fillna(processed_df[col]).median()
                elif method == 'mode':
                    processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
                elif method == 'zero':
                    processed.df[col] = processed.df[col].fillna(0)
                elif method == 'forward':
                    processed_df[col] = processed_df[col].fillna(method='ffill')
                elif method == 'backward':
                    processed_df[col] = processed_df[col].fillna(method='bfill')
                    
    if 'handle_outliers' in config and config['handle_outliers']:
        for col, method in config['handle_outliers'].items():
            if col in processed_df.columns and processed_df[col].dtype in [np.float64, np.int64]:
                if method == 'clip':
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_df[col] = processed_df[col].clip(lower_bound, upper_bound)
                elif method == 'remove':
                    Q1 = processed_df[col].quantile(0.25)
                    Q3 = processed_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_df = processed_df[(processed_df[col] >= lower_bound) & (processed_df[col] <= upper_bound)]
                    
    if 'scaling' in config and config['scaling']:
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        
        scaling_method = config['scaling']['method']
        columns = config['scaling'].get('columns', numeric_cols)
        
        valid_cols = [col for col in columns if col in numeric_cols]
        
        if valid_cols:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                return processed_df
            
            processed_df[valid_cols] = scaler.fit_transform(processed_df[valid_cols])
            
    if 'feature_engineering' in config and config['feature_engineering']:
        
        for col, features in config['feature_engineering'].items():
            if col in processed_df.columns:
                
                if 'datetime_features' in features and pd.api.types.is_datetime64_any_dtype(processed_df[col]):
                    if 'year' in features['datetime_features']:
                        processed_df[f'{col}_year'] = processed_df[col].dt.year
                    
                    if 'month' in features['datetime_features']:
                        processed_df[f'{col}_month'] = processed_df[col].dt.month
                    
                    if 'day' in features['datetime_features']:
                        processed_df[f'{col}_day'] = processed_df[col].dt.day
                    
                    if 'weekday' in features['datetime_features']:
                        processed_df[f'{col}_weekday'] = processed_df[col].dt.weekday
                    
                    if 'quarter' in features['datetime_features']:
                        processed_df[f'{col}_quarter'] = processed_df[col].dt.quarter
                    
                    if 'is_weekend' in features['datetime_features']:
                        processed_df[f'{col}_is_weekend'] = (processed_df[col].dt.weekday >= 5).astype(int)
                if 'binning' in features and processed_df[col].dtype in [np.float64, np.int64]:
                    n_bins = features['binning'].get('n_bins', 5)
                    labels = features['binning'].get('labels', None)
                    processed_df[f'{col}_bin'] = pd.cut(
                        processed_df[col],
                        bins=n_bins,
                        labels=labels
                    )
                    
                if 'text_features' in features and pd.api.types.is_string_dtype(processed_df[col]):
                    if 'length' in features['text-features']:
                        processed_df[f'{col}_length'] = processed_df[col].str.len()
                    
                    if 'word_count' in features['text_features']:
                        processed_df[f'{col}_word_count'] = processed_df[col].str.split().str.len()
                    
                    if 'contains' in features['text_features']:
                        processed_df[new_col] = processed_df[col].str.contains(item, case=False).astype(int)
                        
    if 'encoding' in config and config['encoding']:
        for col, method in config['encoding'].items():
            if col in processed_df.columns:
                if method == 'onehot':
                    
                    dummies = pd.get_dummies(processed_df[col], prefix=col)
                    processed_df = pd.concat([processed_df, dummies], axis=1)
                    processed_df = processed_df.drop(col,axis=1)
                elif method == 'label':
                    
                    processed_dt[col] = processed_df[col.astype('category')].cat.codes
                    
    if 'drop_columns' in config and config['drop_columns']:
        cols_to_drop = [col for col in config['drop_columns'] if col in processed_df.columns]
        processed_df = processed_df.drop(cols_to_drop, axis=1)
        
    return processed_df

