"""
競馬AI メインスクリプト

データ読み込み → 前処理 → 学習 → 評価 の一連のパイプラインを実行する。
"""

import random
import warnings
import pickle

import numpy as np
import optuna
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from src import Listwise
from src.config import FIELD_MAPPING
from preprocessing import (
    create_label_gain,
    df_big_past_processing,
    df_end_processing,
    df_first_processing,
    past_level,
)
from train import first_train, stacking, test
from utils import save_csv


# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #

SEED      = 1
FIELD     = 9           # 開催場所番号（中京）
FIELD_NAME = 'chukyo'
CSV_PATH  = './csv/chukyo_2012-2024.csv'
FILE_NUM  = 1           # アンサンブル seed 数


# ------------------------------------------------------------------ #
# エントリーポイント
# ------------------------------------------------------------------ #

def main():
    warnings.simplefilter('ignore')
    optuna.logging.disable_default_handler()

    random.seed(SEED)
    np.random.seed(SEED)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    # --- データ読み込み ---
    df = pd.read_csv(CSV_PATH, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['場所'] = FIELD
    df = df.reset_index(drop=True)

    # --- 前処理 ---
    df       = df_first_processing(df, FIELD_NAME, mode='学習')
    df_all   = df_big_past_processing(df, FIELD_NAME, FIELD)
    df_all   = past_level(df_all, mode='学習')
    df_all   = df_end_processing(df_all, mode='学習')

    save_csv(f'./csv/df_all_{FIELD_NAME}_2025.csv', df_all)

    # --- ラベル生成 ---
    df_all, gain_list = create_label_gain(df_all)

    # --- Listwise 加工 ---
    df_lw = Listwise.inversion(df_all)
    df_lw = Listwise.append_col(df_lw)
    df_lw = Listwise.add_relative_features(df_lw)

    feature_cols = [
        col for col in df_lw.columns
        if col not in [
            'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank',
            'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ',
            '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob',
            'is_win', 'win_prob_by_rank',
        ]
    ]

    # 父馬のターゲットエンコーディング（fold 0 を使用）
    pkl_path = f'./pickle-dict/sire_dict_{FIELD_NAME}_fold0.pkl'
    with open(pkl_path, 'rb') as f:
        sire_mapping = pickle.load(f)
    df_lw['父馬_te'] = df_lw['父馬'].map(sire_mapping).fillna(-1)

    # スケーリング
    scaler = StandardScaler()
    df_lw[Listwise.scale_cols] = scaler.fit_transform(df_lw[Listwise.scale_cols])
    df_lw = Listwise.fill_nan(df_lw, feature_cols)

    # カテゴリ列数値化
    df_lw = Listwise.race_feature(df_lw)
    cat_list = Listwise.embedding_cols + Listwise.context_cat_cols

    # --- 学習 ---
    y_test_id, test_list = first_train(df_lw, feature_cols, cat_list, gain_list, FILE_NUM)

    # --- 評価 ---
    test(test_list, y_test_id, FILE_NUM)

    # --- スタッキング（必要に応じてコメントを外す）---
    # stacking(y_test_id, gain_list, FILE_NUM)


if __name__ == '__main__':
    main()
