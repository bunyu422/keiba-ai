# === 学習・データ設定（ListNet 用） ===

import pandas as pd
import numpy as np
import torch

# === CV / 学習ハイパーパラメータ ===
n_splits = 5           # クロスバリデーション分割数
num_epochs = 1000      # 最大エポック数
batch_size = 1          # 1レースずつバッチ処理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === コンテキスト（レース全体共通）特徴量 ===
context_cat_cols = ['フィールド', '馬場']  # レース単位カテゴリ特徴
context_num_cols = ['距離']                # レース単位数値特徴

# === 標準化対象の数値特徴量 ===
scale_cols = ['フィールド適性スコア', '馬場適性スコア', '距離適性スコア', '間隔', '1クラス差', '1ペース差', '父馬_te', '騎手_te',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z', '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z']

# === 反転対象カラム（小さいほど良い値 → 符号反転）===
inversion_cols = []

# === モデル入力から除外するが後で参照する共通カラム ===
common_cols = ['場所','距離','フィールド','馬場','騎手','馬番','1距離','1場所','1フィールド']

# === カテゴリ特徴量（embedding 入力）===
feature_category = ['父馬', '騎手', '性', '齢']
embedding_cols = feature_category

# === データセットのグループ・ターゲット ===
group_col = 'レースID'
target_col = '着順'
feature_cols = []

# === データファイルパス ===
field = 'hanshin'
csv_path = f'./csv/df_all_{field}_2025_add.csv'

# === ペアワイズ損失探索用 ===
pairwise_list = ['logistic']
weight_mode_list = ['roi']
