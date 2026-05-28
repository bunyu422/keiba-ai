"""
汎用ユーティリティ
- CSV 保存
- Pickle 辞書の作成・読み込み
- 型変換ヘルパー
"""

import pickle
import re

import numpy as np
import pandas as pd


def save_csv(path, df):
    """DataFrame を CSV に保存する。"""
    df.to_csv(path, na_rep='NaN')


def create_unique_pickle(series, file_path):
    """
    シリーズのユニーク値から整数マッピング辞書を作成して Pickle に保存する。

    Returns
    -------
    dict
        {値: 連番} の辞書
    """
    mapping = {v: i for i, v in enumerate(series.unique(), start=1)}
    with open(file_path, 'wb') as f:
        pickle.dump(mapping, f)
    return mapping


def load_pickle(file_path):
    """Pickle ファイルを読み込んで返す。"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def convert_to_float(df):
    """
    数値に変換可能な列をすべて float に変換した DataFrame を返す。
    変換できない列はそのまま残す。
    """
    df = df.copy()
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors='raise')
            if pd.api.types.is_numeric_dtype(converted):
                df[col] = converted.astype(float)
        except Exception:
            pass
    return df


def time_to_seconds(s):
    """
    タイム文字列を秒数（float）に変換する。

    Examples
    --------
    '1:34.5' → 94.5
    """
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if not re.match(r'^\d+:\d+(\.\d+)?$', s):
        return np.nan
    m, sec = s.split(':')
    return int(m) * 60 + float(sec)


def target_encoding(df, column, target):
    """
    5-fold ターゲットエンコーディングを行い、変換後の DataFrame を返す。

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        エンコード対象列名
    target : str
        目的変数列名

    Returns
    -------
    pd.DataFrame
    """
    chunk = int(len(df) / 5) + 1
    folds = [df.iloc[i:i+chunk, :] for i in range(0, len(df), chunk)]
    result = pd.DataFrame()

    for i in range(len(folds)):
        val_fold = folds[i].copy()
        train_folds = pd.concat([folds[j] for j in range(len(folds)) if j != i], axis=0)
        mapping = train_folds.groupby(column)[target].mean().to_dict()
        val_fold[column] = pd.to_numeric(val_fold[column].map(mapping), errors='coerce')
        result = pd.concat([result, val_fold], axis=0)

    return result
