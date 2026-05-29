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

from src.common import transform
from src.common.config import FIELD_MAPPING
from src.common.preprocessing import (
    create_label_gain,
    df_big_past_processing,
    df_end_processing,
    df_first_processing,
    past_level,
)
from src.common.utils import save_csv
from src.lightgbm.train import first_train, stacking, test


# ------------------------------------------------------------------ #
# 設定
# ------------------------------------------------------------------ #

SEED      = 1
FIELD     = 9           # 開催場所番号（中京）
FIELD_NAME = 'chukyo'
CSV_PATH  = './csv/chukyo_2012-2024.csv'
FILE_NUM  = 1           # アンサンブル seed 数

# === 特徴量カラムの定義（ListNet と同一スキーマ）===
context_cat_cols = ['フィールド', '馬場']
context_num_cols = ['距離']

scale_cols = ['フィールド適性スコア', '馬場適性スコア', '距離適性スコア', '間隔', '1クラス差', '1ペース差', '父馬_te', '騎手_te',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z',
              '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z',
              '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z',
              '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z',
              '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z',
              '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z',
              'past_score_mean', 'past_score_max', 'past_score_min', 'past_score_sum', 'past_score_ewm',
              '同距離過去率', '同場所過去率',
]

inversion_cols = []

feature_category = ['父馬', '騎手', '性', '齢']

embedding_cols = feature_category


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

    # --- 変換 ---
    df_lw = transform.inversion(df_all, inversion_cols)

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
    df_lw[scale_cols] = scaler.fit_transform(df_lw[scale_cols])
    df_lw = transform.fill_nan(df_lw, feature_cols)

    # カテゴリ列数値化
    df_lw = transform.race_feature(df_lw, feature_category, context_cat_cols)
    cat_list = embedding_cols + context_cat_cols

    # --- 学習 ---
    y_test_id, test_list = first_train(df_lw, feature_cols, cat_list, gain_list, FILE_NUM)

    # --- 評価 ---
    test(test_list, y_test_id, FILE_NUM)

    # --- スタッキング（必要に応じてコメントを外す）---
    # stacking(y_test_id, gain_list, FILE_NUM)


if __name__ == '__main__':
    main()
