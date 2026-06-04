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
              '距離グループ_父馬_te', '騎手_距離_te', '騎手_フィールド_te',
              '間隔クラス', '前走後3F_レース内順位', '馬体重_trend_slope',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z',               '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z',
              # BEGIN: 中京完成時の追加15列（past_score_mean/max/min/sum/ewm の rank/rel/z）
              'past_score_mean_rank', 'past_score_mean_rel', 'past_score_mean_z',
              'past_score_max_rank', 'past_score_max_rel', 'past_score_max_z',
              'past_score_min_rank', 'past_score_min_rel', 'past_score_min_z',
              'past_score_sum_rank', 'past_score_sum_rel', 'past_score_sum_z',
              'past_score_ewm_rank', 'past_score_ewm_rel', 'past_score_ewm_z',
              # END: 追加15列
]

# === 反転対象カラム（小さいほど良い値 → 符号反転）===
inversion_cols = []

# === モデル入力から除外するが後で参照する共通カラム ===
common_cols = ['場所','距離','フィールド','馬場','騎手','馬番','1距離','1場所','1フィールド','距離グループ']

# === カテゴリ特徴量（embedding 入力）===
feature_category = ['父馬', '騎手', '性', '齢']
embedding_cols = feature_category

# === データセットのグループ・ターゲット ===
group_col = 'レースID'
target_col = '着順'
feature_cols = []

# === データファイルパス ===
field = 'chukyo'
csv_path = f'./csv/df_all_{field}_2025_add.csv'

# === ペアワイズ損失探索用 ===
pairwise_list = [
    # 'logistic',
    'squared_hinge',
]
weight_mode_list = [
    # 'roi',
    'ev_i',
]

# ============================================================ #
# 過去の会場別設定（git history からのメモ）
# 手動で field・seed・target_col 等を書き換えて運用していた
# ============================================================ #
#
#  項目         | 東京 (08bf0c2)              | 阪神 (7086c67)              | 中京 (79eba9d)
#  ────────────┼──────────────────────────────┼─────────────────────────────┼─────────────────────────────
#  field       | 'tokyo'                      | 'hanshin'                   | 'chukyo'
#  seed        | 22                           | 22                          | 4
#  target_col  | 'smooth_rel'                 | '着順'                      | '着順'
#  common_cols | []                           | ['場所','距離','フィール    | ['場所','距離','フィール
#              |                              | ド','馬場','騎手','馬番',   | ド','馬場','騎手','馬番',
#              |                              | '1距離','1場所','1フィー    | '1距離','1場所','1フィー
#              |                              | ルド']                      | ルド']
#  scale_cols  | past_score_mean/max/min/     | past_score_mean/max/min/    | past_score_mean/max/min/
#              | sum/ewm の _rank/_rel/_z     | sum/ewm の _rank/_rel/_z    | sum/ewm の _rank/_rel/_z
#              | あり                         | なし                        | あり
# ============================================================ #
