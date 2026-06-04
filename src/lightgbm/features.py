"""
LightGBM 用特徴量エンジニアリング関数群。

src/listwise/features.py から LightGBM に必要な関数のみ抽出・独立させた。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.common.transform import set_seed, fill_nan


def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df


def target_encoding(df, col, target, n_splits=5, alpha=20, random_state=42):
    df = df.copy()
    df[col + "_te"] = np.nan
    global_mean = df[target].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(df):
        train_data = df.iloc[train_idx]
        stats = train_data.groupby(col)[target].agg(['mean', 'count'])
        stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)
        mapping = stats['smooth'].to_dict()
        val_values = df.iloc[val_idx][col]
        df.iloc[val_idx, df.columns.get_loc(col + "_te")] = val_values.map(mapping).fillna(global_mean)
    full_stats = df.groupby(col)[target].agg(['mean', 'count'])
    full_stats['smooth'] = (full_stats['mean'] * full_stats['count'] + alpha * global_mean) / (full_stats['count'] + alpha)
    full_mapping = full_stats['smooth'].to_dict()
    return df, full_mapping


def _distance_group(dist):
    if dist <= 1400:
        return 0
    elif dist <= 1800:
        return 1
    elif dist <= 2400:
        return 2
    else:
        return 3


def add_distance_group(df):
    df = df.copy()
    df['距離グループ'] = df['距離'].apply(_distance_group)
    return df


def add_interval_class(df):
    df = df.copy()
    def _classify(x):
        if pd.isna(x) or x == 0:
            return 0
        elif x <= 2:
            return 1
        elif x <= 4:
            return 2
        elif x <= 8:
            return 3
        else:
            return 4
    df['間隔クラス'] = df['間隔'].apply(_classify)
    return df


def add_last3f_race_rank(df):
    df = df.copy()
    df['前走後3F_レース内順位'] = df.groupby('レースID')['1後3F'].rank(ascending=True, method='dense')
    return df


def add_weight_trend_slope(df, n_past=5):
    df = df.copy()
    weight_cols = [f'{i}馬体重' for i in range(1, n_past + 1)]
    def _calc_slope(row):
        weights = row[weight_cols].values.astype(float)
        mask = ~np.isnan(weights)
        if mask.sum() < 2:
            return 0.0
        x = np.arange(n_past)[mask]
        slope, _ = np.polyfit(x, weights[mask], 1)
        return slope
    df['馬体重_trend_slope'] = df.apply(_calc_slope, axis=1)
    return df


def add_interaction_te_fold(train_df, val_df, test_df, df_2025, group_cols, target='着順', alpha=20):
    col_name = '_'.join(group_cols) + '_te'
    global_mean = train_df[target].mean()
    train_df = train_df.copy()

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df[col_name] = np.nan
    for train_cv_idx, val_cv_idx in kf.split(train_df):
        cv_train = train_df.iloc[train_cv_idx]
        cv_val = train_df.iloc[val_cv_idx]
        stats = cv_train.groupby(group_cols)[target].agg(['mean', 'count'])
        stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)
        cv_mapping = stats['smooth'].to_dict()
        def _cv_map(row):
            return cv_mapping.get(tuple(row[g] for g in group_cols), global_mean)
        train_df.iloc[val_cv_idx, train_df.columns.get_loc(col_name)] = cv_val.apply(_cv_map, axis=1).values

    stats = train_df.groupby(group_cols)[target].agg(['mean', 'count'])
    stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)
    mapping = stats['smooth'].to_dict()
    def _map_row(row):
        return mapping.get(tuple(row[g] for g in group_cols), global_mean)

    val_df = val_df.copy()
    test_df = test_df.copy()
    df_2025 = df_2025.copy()
    val_df[col_name] = val_df.apply(_map_row, axis=1)
    test_df[col_name] = test_df.apply(_map_row, axis=1)
    df_2025[col_name] = df_2025.apply(_map_row, axis=1)

    return train_df, val_df, test_df, df_2025, col_name, mapping
