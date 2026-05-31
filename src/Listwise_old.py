import copy
import itertools
import pickle
import random
import warnings
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import Learning
import torch.nn.functional as F
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import entropy, wasserstein_distance
import sys

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 警告を例外に変えてトレースバックを出す
# warnings.filterwarnings("error")

# === 0. ハイパーパラメータ ===
n_splits = 5
num_epochs = 1000
batch_size = 1  # 1レースずつ処理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# common_cols = ['距離', 'フィールド', '馬場', '場所', '出走頭数', '平均クラス', '平均ペース']

# context_cat_cols = ['フィールド', '馬場', '出走頭数', '場所']
# context_num_cols = ['距離', '平均クラス', '平均ペース']

# numeric_diff_cols = ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数',
#                     '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', 
#                     'best着差', 'bestスピード指数', 'best後3F', 'av着差', 'avスピード指数', 'av後3F', '斤量']

# scale_cols = ['1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
#             '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
#             '斤量', '1斤量', '2斤量', '3斤量', '4斤量', '5斤量', '間隔', '1馬体重', '2馬体重', '3馬体重', '4馬体重', '5馬体重',
#             '1体重増減', '2体重増減', '3体重増減', '4体重増減', '5体重増減', '1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数',
#             'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度', '1クラス差', '1ペース差',
#             '父馬_te', '騎手_te']
#             # ]

# inversion_cols = ['人気', '齢', '間隔', '1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
#                 '1人気', '2人気', '3人気', '4人気', '5人気', 
#                 '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
#                 'best着差', 'av着差']

# feature_category = ['性', '父馬', '騎手',
#                     '1場所', '2場所', '3場所', '4場所', '5場所', '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド',
#                     '1距離', '2距離', '3距離', '4距離', '5距離', 
#                     '1馬場', '2馬場', '3馬場', '4馬場', '5馬場','1コーナー通過順', '2コーナー通過順', '3コーナー通過順', '4コーナー通過順', '5コーナー通過順',
#                     '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数', '1馬番', '2馬番', '3馬番', '4馬番', '5馬番']
# diff_category_place = ['1場所変化', '2場所変化', '3場所変化', '4場所変化', '5場所変化']

# diff_category_field = ['1フィールド変化', '2フィールド変化', '3フィールド変化', '4フィールド変化', '5フィールド変化']

context_cat_cols = ['フィールド', '馬場']
context_num_cols = ['距離']

scale_cols = ['フィールド適性スコア', '馬場適性スコア', '距離適性スコア', '間隔', '1クラス差', '1ペース差', '父馬_te', '騎手_te',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z', '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z', 'past_score_mean_rank', 'past_score_mean_rel', 'past_score_mean_z', 'past_score_max_rank', 'past_score_max_rel', 'past_score_max_z', 'past_score_min_rank', 'past_score_min_rel', 'past_score_min_z', 'past_score_sum_rank', 'past_score_sum_rel', 'past_score_sum_z', 'past_score_ewm_rank', 'past_score_ewm_rel', 'past_score_ewm_z']

inversion_cols = []
common_cols = ['場所','距離','フィールド','馬場','騎手','馬番','1距離','1場所','1フィールド'] # 阪神, 中京
# common_cols = [] # 東京，中山

feature_category = ['父馬', '騎手', '性', '齢']

embedding_cols = feature_category

# past_cols = [['1_past_score_rank', '1_past_score_rel', '1_past_score_z', '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z'], ['2_past_score_rank', '2_past_score_rel', '2_past_score_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z'], ['3_past_score_rank', '3_past_score_rel', '3_past_score_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z'], ['4_past_score_rank', '4_past_score_rel', '4_past_score_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z'], ['5_past_score_rank', '5_past_score_rel', '5_past_score_z','5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z']]
# add_cols = ['1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z', '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z']

# past_cols = [['1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化'], ['2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2コーナー通過順', '2後3F', '2馬体重', '2体重増減', '2着差', '2クラス', '2スピード指数', '2距離差', '2場所変化', '2フィールド変化'], ['3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3コーナー通過順', '3後3F', '3馬体重', '3体重増減', '3着差', '3クラス', '3スピード指数', '3距離差', '3場所変化', '3フィールド変化'], ['4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4コーナー通過順', '4後3F', '4馬体重', '4体重増減', '4着差', '4クラス', '4スピード指数', '4距離差', '4場所変化', '4フィールド変化'], ['5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5コーナー通過順', '5後3F', '5馬体重', '5体重増減', '5着差', '5クラス', '5スピード指数', '5距離差', '5場所変化', '5フィールド変化']]
# add_cols = ['1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2コーナー通過順', '2後3F', '2馬体重', '2体重増減', '2着差', '2クラス', '2スピード指数', '2距離差', '2場所変化', '2フィールド変化', '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3コーナー通過順', '3後3F', '3馬体重', '3体重増減', '3着差', '3クラス', '3スピード指数', '3距離差', '3場所変化', '3フィールド変化', '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4コーナー通過順', '4後3F', '4馬体重', '4体重増減', '4着差', '4クラス', '4スピード指数', '4距離差', '4場所変化', '4フィールド変化', '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5コーナー通過順', '5後3F', '5馬体重', '5体重増減', '5着差', '5クラス', '5スピード指数', '5距離差', '5場所変化', '5フィールド変化']
# scale_cols = scale_cols + add_cols

# embedding_cols = feature_category + diff_category_place + diff_category_field

def load_csv(path):
    # 学習データを読み込む
    df = pd.read_csv(path, index_col=0)
    return df

# file_path
###########################モデルごとに変更が必要############################
field = 'chukyo'
# csv_path = f'./csv/df_all_chukyo_2025_add.csv'
csv_path = f'./csv/df_all_{field}_2025_add.csv'
###########################################################################

df = load_csv(csv_path)
group_col = 'レースID'
target_col = '着順'
feature_cols = []
df['is_win'] = (df['着順'] == 1).astype(int)

def compute_roi_stats(df, course_cols, target_cols, win_col='is_win', odds_col='単勝オッズ'):
    """
    df: 集計に使うデータ
    course_cols: ['場所','距離','フィールド','馬場']
    target_cols: ['騎手','馬番','1距離','1場所','1フィールド']
    """
    stats = {}

    for col in target_cols:
        g = (
            df.groupby(course_cols + [col])
            .apply(lambda x: pd.Series({
                "n": len(x),
                "payout": (x[odds_col] * x[win_col]).sum(),
            }))
            .reset_index()
        )
        
        g["roi"] = g["payout"] / (g["n"] * 100) # 正しい ROI

        stats[col] = g[course_cols + [col, "roi"]]

    return stats

def merge_roi_features(df, stats, course_cols, target_cols):
    add_cols = []
    df_out = df.copy()
    for col in target_cols:
        df_out = df_out.merge(
            stats[col],
            on=course_cols + [col],
            how='left',
            suffixes=('', f'_roi_{col}')
        )
        df_out[f'roi_{col}'] = df_out['roi'].fillna(df['roi'].mean() if 'roi' in df else 1.0)
        df_out = df_out.drop(columns=['roi'])
        add_cols.append(f'roi_{col}')
    return df_out, add_cols

def build_train_features(df_train):
    course_cols = ['場所','距離','フィールド','馬場']
    target_cols = ['騎手','馬番','1距離','1場所','1フィールド']

    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    df_train = df_train.copy()
    df_train['roi_feat_dummy'] = np.nan  # ダミー

    parts = []

    for (idx_a, idx_b) in kf.split(df_train):
        A = df_train.iloc[idx_a]
        B = df_train.iloc[idx_b]

        # A で統計 → B に適用
        stats_A = compute_roi_stats(A, course_cols, target_cols)
        B2, _ = merge_roi_features(B, stats_A, course_cols, target_cols)

        # B で統計 → A に適用
        stats_B = compute_roi_stats(B, course_cols, target_cols)
        A2, _ = merge_roi_features(A, stats_B, course_cols, target_cols)

        parts.append(A2)
        parts.append(B2)

    df_out = pd.concat(parts).sort_index()
    return df_out

def apply_val_test_features(df_train, df_eval):
    course_cols = ['場所','距離','フィールド','馬場']
    target_cols = ['騎手','馬番','1距離','1場所','1フィールド']

    stats_train = compute_roi_stats(df_train, course_cols, target_cols)
    df_eval_out, add_cols = merge_roi_features(df_eval, stats_train, course_cols, target_cols)
    return df_eval_out, add_cols

# def append_col(df):
#     df = df.sort_values(['レースID', '馬番'])
#     df = add_history_features(df)  # 元々の履歴特徴追加関数

#     df['best後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).min(axis=1)
#     df['av後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).mean(axis=1)
#     # ascending=True → 値が小さいほど小さい順位
#     # → 大きい数値ほど順位が大きくなる（方向性統一）
#     df['best_speed_rank_num'] = df.groupby('レースID')['bestスピード指数'].rank(method='min', ascending=True)
#     df['av_speed_rank_num'] = df.groupby('レースID')['avスピード指数'].rank(method='min', ascending=True)
#     df['best後3F_rank_num'] = df.groupby('レースID')['best後3F'].rank(method='min', ascending=False)
#     df['av後3F_rank_num'] = df.groupby('レースID')['av後3F'].rank(method='min', ascending=False)

#     df['best_speed_rank_cat'] = df['best_speed_rank_num']
#     df['av_speed_rank_cat'] = df['av_speed_rank_num']
#     df['best後3F_rank_cat'] = df['best後3F_rank_num']
#     df['av後3F_rank_cat'] = df['av後3F_rank_num']

#     feature_cols.append('best後3F')
#     feature_cols.append('av後3F')
#     feature_cols.append('best_speed_rank_num')
#     feature_cols.append('av_speed_rank_num')
#     feature_cols.append('best後3F_rank_num')
#     feature_cols.append('av後3F_rank_num')


#     scale_cols.append('best後3F')
#     scale_cols.append('av後3F')
#     scale_cols.append('best_speed_rank_num')
#     scale_cols.append('av_speed_rank_num')
#     scale_cols.append('best後3F_rank_num')
#     scale_cols.append('av後3F_rank_num')

#     feature_category.append('best_speed_rank_cat')
#     feature_category.append('av_speed_rank_cat')
#     feature_category.append('best後3F_rank_cat')
#     feature_category.append('av後3F_rank_cat')
    
    
    
#     return df

# def append_col(df):
#     df = add_history_features(df)  # 元々の履歴特徴追加関数
#     df = add_course_bias_features(df)

#     # --- 過去後3F / スピード指数の min / mean ---
#     df['best後3F'] = df.loc[:, ['1後3F','2後3F','3後3F','4後3F','5後3F']].astype(float).min(axis=1)
#     df['av後3F'] = df.loc[:, ['1後3F','2後3F','3後3F','4後3F','5後3F']].astype(float).mean(axis=1)

#     df['best_speed_rank_num'] = df.groupby('レースID')['bestスピード指数'].rank(method='min', ascending=True)
#     df['av_speed_rank_num'] = df.groupby('レースID')['avスピード指数'].rank(method='min', ascending=True)
#     df['best後3F_rank_num'] = df.groupby('レースID')['best後3F'].rank(method='min', ascending=False)
#     df['av後3F_rank_num'] = df.groupby('レースID')['av後3F'].rank(method='min', ascending=False)

#     # --- 過去3～5走の傾き / 標準偏差（Trend / 安定度） ---
#     df['speed_std_5r'] = df.loc[:, ['1スピード指数','2スピード指数','3スピード指数','4スピード指数','5スピード指数']].std(axis=1)
#     df['後3F_std_5r'] = df.loc[:, ['1後3F','2後3F','3後3F','4後3F','5後3F']].std(axis=1)
#     df['着差_std_5r'] = df.loc[:, ['1着差','2着差','3着差','4着差','5着差']].std(axis=1)

#     # --- Trend（直近3走の回帰傾き） ---
#     for col_base, new_col in [('1スピード指数','speed_trend_3r'), ('1後3F','後3F_trend_3r'), ('1着差','着差_trend_3r'), ('1人気','人気_trend_3r')]:
#         df[new_col] = df[[f'{i}{col_base[1:]}' for i in range(1,4)]].astype(float).apply(lambda x: np.polyfit(range(3), x, 1)[0], axis=1)

#     # --- レース内相対差 / 相対順位 ---
#     df['speed_diff_best'] = df.groupby('レースID')['bestスピード指数'].transform(lambda x: x.max() - x)
#     df['speed_diff_av'] = df.groupby('レースID')['avスピード指数'].transform(lambda x: x.mean() - x)
#     df['後3F_diff_best'] = df.groupby('レースID')['best後3F'].transform(lambda x: x.min() - x)
#     df['後3F_diff_av'] = df.groupby('レースID')['av後3F'].transform(lambda x: x.mean() - x)
#     df['相対順位_best'] = df.groupby('レースID')['bestスピード指数'].rank(method='min', ascending=False)
#     df['相対順位_av'] = df.groupby('レースID')['avスピード指数'].rank(method='min', ascending=False)

#     # --- 間隔 / 斤量差分 ---
#     # df['間隔_mean_3r'] = df[['1間隔','2間隔','3間隔']].astype(float).mean(axis=1)
#     # df['間隔_diff_last'] = df['間隔'].astype(float) - df['間隔'].astype(float).shift(1)
#     df['斤量_diff_last'] = df['斤量'].astype(float) - df['1斤量'].astype(float)

#     # --- 距離適性 / 期待値 ---
#     # 例として単純正規化
#     df['距離適性_score'] = df['bestスピード指数'] / df.groupby('距離')['bestスピード指数'].transform('max')
#     df['期待値'] = (1/df['オッズ']).fillna(0)  # 単勝期待値の簡易計算
#     # df['順位正規化'] = df.groupby('レースID')['着順'].rank(method='min', ascending=True) / df.groupby('レースID')['着順'].transform('max')

#     # --- カテゴリ特徴追加 ---
#     df['馬×距離'] = df['馬番'].astype(str) + '_' + df['距離'].astype(str)
#     df['馬×騎手'] = df['馬番'].astype(str) + '_' + df['騎手'].astype(str)
#     df['馬×コース'] = df['馬番'].astype(str) + '_' + df['場所'].astype(str)
#     df['馬×斤量帯'] = df['馬番'].astype(str) + '_' + pd.cut(df['斤量'], bins=[48,50,52,54,56,58,60], labels=False).astype(str)
#     df['best_speed_rank_cat'] = df['best_speed_rank_num']
#     df['av_speed_rank_cat'] = df['av_speed_rank_num']
#     df['best後3F_rank_cat'] = df['best後3F_rank_num']
#     df['av後3F_rank_cat'] = df['av後3F_rank_num']

#     # --- 新しい特徴列を feature_cols に追加 ---
#     new_cols = [
#         'best後3F','av後3F','best_speed_rank_num','av_speed_rank_num','best後3F_rank_num','av後3F_rank_num',
#         'speed_std_5r','後3F_std_5r','着差_std_5r','speed_trend_3r','後3F_trend_3r','着差_trend_3r','人気_trend_3r',
#         'speed_diff_best','speed_diff_av','後3F_diff_best','後3F_diff_av','相対順位_best','相対順位_av',
#         '斤量_diff_last','距離適性_score','期待値',
#         '馬×距離','馬×騎手','馬×コース','馬×斤量帯','同場所過去率','同距離過去率'
#     ]
#     feature_cols.extend(new_cols)
#     scale_cols.extend([
#         'best後3F','av後3F','best_speed_rank_num','av_speed_rank_num','best後3F_rank_num','av後3F_rank_num',
#         'speed_std_5r','後3F_std_5r','着差_std_5r','speed_trend_3r','後3F_trend_3r','着差_trend_3r','人気_trend_3r',
#         'speed_diff_best','speed_diff_av','後3F_diff_best','後3F_diff_av',
#         '斤量_diff_last','距離適性_score','期待値','同場所過去率','同距離過去率'
#     ])
#     feature_category.extend(['馬×距離','馬×騎手','馬×コース','馬×斤量帯','best_speed_rank_cat','av_speed_rank_cat','best後3F_rank_cat','av後3F_rank_cat',
#                              '同距離過去数','同距離過去3着内数','同場所過去数','同場所過去3着内数'])

#     return df

# def add_course_bias_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     中山芝・ダートの距離別コース傾向を特徴量化して追加
#     """
#     df = df.copy()

#     # --- 基本 ---
#     df['枠'] = ((df['馬番'] - 1) // 2 + 1).astype(int)
#     df['is_inner'] = (df['枠'] <= 3).astype(int)
#     df['is_middle'] = ((df['枠'] >= 4) & (df['枠'] <= 6)).astype(int)
#     df['is_outer'] = (df['枠'] >= 7).astype(int)
#     df['pos_ratio'] = df['1コーナー通過順'] / df['出走頭数']

#     # --- 持久力・上がり系 ---
#     if all(col in df.columns for col in ['1スピード指数', '1ペース差']):
#         df['持久力指数'] = df['1スピード指数'] - df['1ペース差'].abs()
#     else:
#         df['持久力指数'] = np.nan

#     past_3f_cols = [c for c in df.columns if '後3F' in c]
#     df['av_上がり'] = df[past_3f_cols].mean(axis=1) if past_3f_cols else np.nan

#     # --- 中山コース別傾向 ---
#     def course_profile(row):
#         f, d = row['フィールド'], row['距離']
#         pos, inner, mid, outer = row['pos_ratio'], row['is_inner'], row['is_middle'], row['is_outer']
#         stamina, agari = row['持久力指数'], row['av_上がり']

#         score = np.nan

#         if f == 1:
#             # --- 中山芝1200m ---
#             if d == 1200:
#                 score = 0.6 * inner + 0.4 * (1 - pos)

#             # --- 中山芝1600m ---
#             elif d == 1600:
#                 score = 0.4 * (1 - abs(pos - 0.4)) + 0.3 * stamina + 0.3 * agari

#             # --- 中山芝1800m ---
#             elif d == 1800:
#                 score = 0.5 * (1 - pos) + 0.3 * outer + 0.2 * stamina

#             # --- 中山芝2000m ---
#             elif d == 2000:
#                 score = (
#                     0.5 * (1 - pos) +           # 前有利
#                     0.3 * mid +                 # 真ん中〜外がベター
#                     0.2 * (1 - inner) -         # 内枠はややマイナス
#                     0.2 * (row['人気'] / 18.0)  # 人気先行馬は減点（上級クラスは逃げ苦戦）
#                 )

#             # --- 中山芝2200m ---
#             elif d == 2200:
#                 score = (
#                     0.5 * (1 - pos) +       # 前有利
#                     0.3 * inner +           # 内枠（特に1枠）が良い
#                     0.2 * (1 / (row['人気'] + 1))
#                 )

#             # --- 中山芝2500m ---
#             elif d == 2500:
#                 score = (
#                     0.4 * (1 - abs(pos - 0.4)) +  # 好位〜中位
#                     0.2 * (1 - row['pos_std']) +  # 安定性
#                     0.2 * agari +                 # 上がり性能
#                     0.2 * ((row['枠'] == 3) | (row['枠'] == 8))  # 3枠・8枠優位
#                 )

#             # --- 中山芝3600m ---
#             elif d == 3600:
#                 score = (
#                     0.4 * (1 - abs(pos - 0.35)) + # 好位有利
#                     0.3 * mid +                   # 中枠妙味
#                     0.3 * stamina                 # 長く脚を使えるタイプ
#                 )

#         elif f == 2:
#             # --- 中山ダート1200m ---
#             if d == 1200:
#                 score = (
#                     0.6 * (1 - pos) +           # 前有利
#                     0.3 * ((row['枠'] == 3) | (row['枠'] == 5)) +  # 3・5枠狙い目
#                     0.1 * (1 / (row['人気'] + 1))
#                 )

#             # --- 中山ダート1800m ---
#             elif d == 1800:
#                 score = (
#                     0.5 * (1 - pos) +     # 前有利
#                     0.3 * row['is_inner'] +  # 内枠優勢（2枠良）
#                     0.2 * stamina
#                 )

#             # --- 中山ダート2400m・2500m ---
#             elif d in [2400, 2500]:
#                 score = (
#                     0.4 * (1 - pos) +                # 前目
#                     0.3 * ((row['枠'] == 2) | (row['枠'] == 3)) +  # 2・3枠有利
#                     0.2 * stamina +
#                     0.1 * (1 - outer)                # 外枠割引
#                 )

#         return score

    
#     # --- 安定性特徴 ---
#     pos_cols = [c for c in df.columns if 'コーナー通過順' in c]
#     df['pos_std'] = df[pos_cols].std(axis=1) if len(pos_cols) >= 2 else np.nan
#     df['上がり_std'] = df[past_3f_cols].std(axis=1) if len(past_3f_cols) >= 2 else np.nan

#     df['course_profile_score'] = df.apply(course_profile, axis=1)


#     # --- 総合コース適性 ---
#     df['コース適性総合'] = (
#         0.4 * (1 - df['pos_ratio']) +
#         0.2 * (1 - df['pos_std'].fillna(0)) +
#         0.2 * df['course_profile_score'].fillna(0) +
#         0.2 * df['持久力指数'].fillna(0)
#     )

#     # apply のあと
#     df['course_profile_score'] = df['course_profile_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
#     df['コース適性総合'] = df['コース適性総合'].replace([np.inf, -np.inf], np.nan).fillna(0)

#     feature_cols.extend([
#     '枠', 'pos_ratio', '持久力指数', 'av_上がり',
#     'course_profile_score', 'pos_std', '上がり_std', 'コース適性総合'
#     ])

#     scale_cols.extend([
#     '枠', 'pos_ratio', '持久力指数', 'av_上がり',
#     'course_profile_score', 'pos_std', '上がり_std', 'コース適性総合'
#     ])

#     feature_category.extend([
#     '枠', 'is_inner', 'is_middle', 'is_outer'
#     ])

#     return df

def add_history_features(df):
    # 距離関連
    dist_cols = [f"{i}距離" for i in range(1, 6)]
    dist_rank_cols = [f"{i}過去着順" for i in range(1, 6)]

    def safe_to_num(val):
        try:
            return float(val)
        except:
            return np.nan

    def count_same_distance(row):
        return sum(row["距離"] == row[col] for col in dist_cols)

    def count_top3_same_distance(row):
        counts = [
            safe_to_num(row[dist_rank_cols[i]])
            for i in range(5) if row["距離"] == row[dist_cols[i]]
        ]
        return sum((not np.isnan(r)) and (r <= 3) for r in counts)

    df["同距離過去数"] = df.apply(count_same_distance, axis=1)
    df["同距離過去3着内数"] = df.apply(count_top3_same_distance, axis=1)
    df['同距離過去率'] = df['同距離過去3着内数'] / (df['同距離過去数'] + 1e-12)

    # 場所関連
    loc_cols = [f"{i}場所" for i in range(1, 6)]
    loc_rank_cols = [f"{i}過去着順" for i in range(1, 6)]

    def count_same_location(row):
        return sum(row["場所"] == row[col] for col in loc_cols)

    def count_top3_same_location(row):
        counts = [
            safe_to_num(row[loc_rank_cols[i]])
            for i in range(5) if row["場所"] == row[loc_cols[i]]
        ]
        return sum((not np.isnan(r)) and (r <= 3) for r in counts)

    df["同場所過去数"] = df.apply(count_same_location, axis=1)
    df["同場所過去3着内数"] = df.apply(count_top3_same_location, axis=1)
    df['同場所過去率'] = df['同場所過去3着内数'] / (df['同場所過去数'] + 1e-12)

    feature_cols.extend(['同距離過去率', '同場所過去率'])
    scale_cols.extend(['同距離過去率', '同場所過去率'])
    feature_category.extend(['同距離過去数', '同距離過去3着内数', '同場所過去数', '同場所過去3着内数'])


    return df

# Nanの処理
def fill_nan(df, cols):
    # 1. NaN を -9999 で埋める
    df[cols] = df[cols].fillna(-9999)

    # 2. 休養フラグをまとめて作る
    # rest_flags = {
    #     f"{i}休養": (df[f"{i}過去着順"] == -9999).astype(int)
    #     for i in range(1, 6)
    # }

    # # 3. 一括で追加（断片化しない）
    # df = pd.concat([df, pd.DataFrame(rest_flags, index=df.index)], axis=1)

    return df

def eval_rank(df):
    # パラメータ設定
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 特徴量と補助変数
    feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', '1休養', '2休養', '3休養', '4休養', '5休養']]
    df['score'] = np.nan
    # print(feature_cols)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df['レースID'].unique())):
        # ランクをラベル化
        train_races = df['レースID'].unique()[train_idx]
        val_races = df['レースID'].unique()[val_idx]

        # データセット分割
        train_data = df[df['レースID'].isin(train_races)]
        val_data = df[df['レースID'].isin(val_races)]

        # モデル読み込み
        with open(f'./pickle-tuner/tokyo_rank_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)

        # OOF予測
        df.loc[val_data.index, 'score'] = model.predict(val_data[feature_cols], num_iteration=model.best_iteration)

def race_feature(df):
    # レースごとの特徴としてまとめる（pandasでの例）
    category = feature_category + context_cat_cols
    for col in category:
        # 1. カラムをカテゴリ型に変換
        df[col] = df[col].astype('category')

        # 2. 0始まりの整数インデックスに変換
        df[col] = df[col].cat.codes
    
    # for col in diff_category_place:
    #     df[col] = df[col].astype(str) + '->' + df['場所'].astype(str)
    #     # 1. カラムをカテゴリ型に変換
    #     df[col] = df[col].astype('category')

    #     # 2. 0始まりの整数インデックスに変換
    #     df[col] = df[col].cat.codes

    # for col in diff_category_field:
    #     df[col] = df[col].astype(str) + '->' + df['フィールド'].astype(str)
    #     # 1. カラムをカテゴリ型に変換
    #     df[col] = df[col].astype('category')

    #     # 2. 0始まりの整数インデックスに変換
    #     df[col] = df[col].cat.codes
    
    
    return df

def race_feature_train(df):
    """
    Train用: カテゴリ列を整数に変換し、マッピングを作成して返す
    """
    category_mappings = {}  # 各列のカテゴリ→整数の対応を保存

    # 通常カテゴリ列
    category = feature_category + context_cat_cols
    # joblib.dump(category, f"./pickle-dict/category_cols.pkl")
    for col in category:
        df[col] = df[col].astype('category')
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

    # 場所の変化特徴
    # for col in diff_category_place:
    #     df[col] = df[col].astype(str) + '->' + df['場所'].astype(str)
    #     df[col] = df[col].astype('category')
    #     category_mappings[col] = dict(enumerate(df[col].cat.categories))
    #     df[col] = df[col].cat.codes

    # # フィールドの変化特徴
    # for col in diff_category_field:
    #     df[col] = df[col].astype(str) + '->' + df['フィールド'].astype(str)
    #     df[col] = df[col].astype('category')
    #     category_mappings[col] = dict(enumerate(df[col].cat.categories))
    #     df[col] = df[col].cat.codes

    return df, category_mappings

def race_feature_test(df, category_mappings):
    """
    Test用: 学習時の mapping を使ってカテゴリ変換
    未知カテゴリは -1 に置換
    """
    # 通常カテゴリ列
    category = feature_category + context_cat_cols
    for col in category:
        if col in df.columns:
            df[col] = df[col].astype(str)
            inv_map = {v:k for k,v in category_mappings[col].items()}
            # inv_map のキー型を確認
            key_type = type(list(inv_map.keys())[0])
            # print(inv_map)
            # train にない値は 0 に置き換える
            n_train_categories = len(category_mappings[col])
            df[col] = df[col].map(lambda x: inv_map.get(key_type(x), n_train_categories)).astype(int)

    # 場所の変化特徴
    # for col in diff_category_place:
    #     if col in df.columns:
    #         df[col] = df[col].astype(str) + '->' + df['場所'].astype(str)
    #         inv_map = {v:k for k,v in category_mappings[col].items()}
    #         # inv_map のキー型を確認
    #         key_type = type(list(inv_map.keys())[0])
    #         # train にない値は 0 に置き換える
    #         n_train_categories = len(category_mappings[col])
    #         df[col] = df[col].map(lambda x: inv_map.get(key_type(x), n_train_categories)).astype(int)

    # # フィールドの変化特徴
    # for col in diff_category_field:
    #     if col in df.columns:
    #         df[col] = df[col].astype(str) + '->' + df['フィールド'].astype(str)
    #         inv_map = {v:k for k,v in category_mappings[col].items()}
    #         # inv_map のキー型を確認
    #         key_type = type(list(inv_map.keys())[0])
    #         # train にない値は 0 に置き換える
    #         n_train_categories = len(category_mappings[col])
    #         df[col] = df[col].map(lambda x: inv_map.get(key_type(x), n_train_categories)).astype(int)

    return df

# スケールの方向をそろえる
def inversion(df):
    for col in inversion_cols:
        df[col] = -df[col]
    return df


# ランク予測
# eval_rank(df)

# === 1. データの前提 ===
# 1着の馬だけ 1、それ以外 0 の one-hot ターゲットを作成
# df['win_prob'] = (df['着順'] == 1).astype(float)

# rankから経験的勝率を計算する
# df['pred_rank'] = df.groupby('レースID')['score'].rank(method='first', ascending=False)

# 2. 実着順が1着（勝利）かどうかのフラグを作成（仮に '着順' カラムがあると仮定）


# 3. 予測順位ごとに勝率を集計
# 出走頭数ビン

# rank_winrate = df.groupby('pred_rank')['is_win'].mean().rename('win_prob_by_rank')
# # print(rank_winrate)

# # 4. 各行に予測順位に応じた勝率をマージ
# df = df.merge(rank_winrate, how='left', left_on='pred_rank', right_index=True)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 数値安定化
    return e_x / e_x.sum()


# print(df['win_prob'].head(30))
# df_sorted = df.sort_values(by=['レースID', 'win_prob'], ascending=[True, False])
# print(df_sorted[['レースID', 'win_prob']].head(30))


# 数値列だけ取り出す（NaN処理したい対象列）
# num_cols = df.select_dtypes(include='number').columns.drop('レースID')
# df[num_cols] = df.groupby('レースID')[num_cols].transform(lambda x: x.fillna(x.mean()))
# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# softmax 計算用関数
def softmax_neg_rank(rankings):
    # -着順 を exp にかけることで、着順が良いほど大きな値になる
    x = -rankings
    exp_x = np.exp(x - np.max(x))  # 安定化のため最大値を引く
    return exp_x / np.sum(exp_x)

# def add_relative_features(df, race_id_col='レースID'):
#     """
#     df: pandas.DataFrame 元データ（差分特徴を追加したいもの）
#     numeric_cols: list[str] 差分を計算したい数値特徴のカラム名リスト
#     race_id_col: str レースIDのカラム名
    
#     戻り値: 差分特徴を追加したDataFrame
#     """
#     df = df.copy()
    
#     # レースごとにグループ化
#     grouped = df.groupby(race_id_col)
    
#     for col in numeric_diff_cols:
#         # レース内平均との差分
#         mean_col = f'{col}_diff_mean'
#         df[mean_col] = df[col] - grouped[col].transform('mean')
        
#         # レース内最小値との差分
#         min_col = f'{col}_diff_min'
#         df[min_col] = df[col] - grouped[col].transform('min')

#         feature_cols.append(mean_col)
#         feature_cols.append(min_col)
    
#     return df

# レースごとに softmax を適用するため、例として 'レースID' 単位で groupby
# df['win_prob'] = df.groupby('レースID')['着順'].transform(softmax_neg_rank)

# === 2. Dataset定義 ===
class RaceDataset(Dataset):
    def __init__(self, X_groups, y_groups,
                 cat_groups, context_num_groups, context_cat_groups, win_groups, payout_groups, win_index_groups, past_groups=None):
        self.X_groups = X_groups
        self.y_groups = y_groups
        self.win_groups = win_groups
        self.payout_groups = payout_groups
        self.cat_groups = cat_groups
        self.context_num_groups = context_num_groups
        self.context_cat_groups = context_cat_groups
        self.win_index_groups = win_index_groups
        # self.past_groups = past_groups

    def __len__(self):
        return len(self.X_groups)

    def __getitem__(self, idx):
        return (
            self.X_groups[idx],         # [頭数, num_features]
            self.y_groups[idx],         # [頭数]
            self.cat_groups[idx],       # [頭数, num_cat_features]
            self.context_num_groups[idx],  # [頭数, num_context_num_features]
            self.context_cat_groups[idx],  # [頭数, num_context_cat_features]
            self.win_groups[idx],       # [頭数]
            self.payout_groups[idx],    # [頭数]
            self.win_index_groups[idx],
            # self.past_groups[idx]
        )


# === 3. モデル定義 ===
class ListNet2(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding_sizes = embedding_sizes                  # ← 追加
        self.context_embedding_sizes = context_embedding_sizes  # ← 追加
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in embedding_sizes
        ])

        # contextのカテゴリ変数embedding（例: フィールド、馬場）
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in context_embedding_sizes
        ])

        # contextの数値特徴 + embの次元
        context_input_dim = context_num_sizes + len(context_embedding_sizes) * emb_dim

        # context（レースごとの共通特徴）処理用のMLP
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 馬特徴の次元をcontextと合わせるプロジェクション
        horse_input_dim = num_features + len(embedding_sizes) * emb_dim
        self.horse_proj = nn.Linear(horse_input_dim, hidden_dim)

        # fcの定義（context加算型）
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim*3, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(48, 1)
        )

        self.rank_gate = nn.Linear(1, hidden_dim)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, 1)
        )

        # # ---- Horse features MLP ----
        # self.horse_fc = nn.Sequential(
        #     nn.Linear(horse_input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )

        # # ---- Residual correction from horse_features only ----
        # self.horse_residual = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )


    def forward(self, x, cat_X, context_num, context_cat, rank_scores=None):
        # 埋め込み層処理
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)  # [頭数, emb_dim_total]

        # 馬特徴（数値 + embedding）
        horse_features = torch.cat([x, emb], dim=1)  # [頭数, num_features + emb_dim_total]

        # ---- ここを変更 ----
        # horse_features = self.horse_fc(horse_features)   # [頭数, hidden_dim]

        # ===== context処理 =====
        # context embedding（1サンプル分）
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)  # [emb_dim * num_context_cat_features]

        # context 数値特徴（1サンプル分）
        context_num = context_num[0]  # [num_context_num_features]

        # 結合 → MLPでhidden_dimに変換
        context_all = torch.cat([context_num, context_emb], dim=0)  # [context_input_dim]
        context_out = self.context_fc(context_all.unsqueeze(0))  # [1, hidden_dim]

        # contextを各馬に複製
        context_expand = context_out.expand(horse_features.size(0), -1)  # [頭数, hidden_dim]

        # ===== contextを初期化ベクトルとして加算 =====
        # horse_featuresとcontext_expandの次元を合わせる必要あり
        if horse_features.size(1) != context_expand.size(1):
            # 線形変換で合わせる
            horse_features = self.horse_proj(horse_features)  # [頭数, hidden_dim]
            pass

        # ====== 順位スコア正規化（オプション） ======
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)  # 高順位ほど1に近い
            norm_rank = norm_rank.unsqueeze(1)  # [頭数, 1]

            # ====== ゲーティング加算 ======
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = horse_features + gate * context_expand
        else:
            # 従来のcontext加算
            horse_features = horse_features + context_expand

        # horse_features = horse_features + context_expand  # 要素ごと加算

        # ====== 交互作用（Hadamard product） ======
        interaction = horse_features * context_expand  # [頭数, hidden_dim]

        # ====== 順位スコア正規化ゲート後 ======
        combined = torch.cat([horse_features, context_expand, interaction], dim=1)

        # ===== 交互作用（3種類） =====
        # mul_interaction = horse_features * context_expand
        # add_interaction = horse_features + context_expand
        # diff_interaction = torch.abs(horse_features - context_expand)

        # # ===== まとめて結合 =====
        # combined = torch.cat([
        #     horse_features,
        #     context_expand,
        #     mul_interaction,
        #     add_interaction,
        #     diff_interaction
        # ], dim=1)

        # ====== 最終出力 ======
        out = self.fc(combined)  # [頭数, 1]

        # ---- Residual（horse_features由来の補正） ----
        # resid = self.horse_residual(horse_features)   # [頭数, 1]
        # out = out + resid

        # --- Residual Connection ---
        # MLPを通した特徴に、元のcombined（馬＋context）をskip接続
        out = out + self.residual_proj(combined)

        # === 🔹 ここを追加：Softmax勝率化 ===
        # scores = out.squeeze(-1)  # [頭数]
        # probs = torch.softmax(scores, dim=0)

        # ===== 最終出力 =====
        return out.squeeze(-1)
        # return probs

class ListNet(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding_sizes = embedding_sizes                  # ← 追加
        self.context_embedding_sizes = context_embedding_sizes  # ← 追加
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in embedding_sizes
        ])

        # contextのカテゴリ変数embedding（例: フィールド、馬場）
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in context_embedding_sizes
        ])

        # contextの数値特徴 + embの次元
        context_input_dim = context_num_sizes + len(context_embedding_sizes) * emb_dim

        # context（レースごとの共通特徴）処理用のMLP
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 馬特徴の次元をcontextと合わせるプロジェクション
        horse_input_dim = num_features + len(embedding_sizes) * emb_dim
        self.horse_proj = nn.Linear(horse_input_dim, hidden_dim)

        # fcの定義（context加算型）
        # self.fc = nn.Sequential(
        #     nn.Linear(hidden_dim*3, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(128, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, 1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 5, 96),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(48, 1)
        )

        self.rank_gate = nn.Linear(1, hidden_dim)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, 1)
        )

        # ---- Horse features MLP ----
        self.horse_fc = nn.Sequential(
            nn.Linear(horse_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ---- Residual correction from horse_features only ----
        self.horse_residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, x, cat_X, context_num, context_cat, rank_scores=None):
        # 埋め込み層処理
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)  # [頭数, emb_dim_total]

        # 馬特徴（数値 + embedding）
        horse_features = torch.cat([x, emb], dim=1)  # [頭数, num_features + emb_dim_total]

        # ---- ここを変更 ----
        horse_features = self.horse_fc(horse_features)   # [頭数, hidden_dim]

        # ===== context処理 =====
        # context embedding（1サンプル分）
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)  # [emb_dim * num_context_cat_features]

        # context 数値特徴（1サンプル分）
        context_num = context_num[0]  # [num_context_num_features]

        # 結合 → MLPでhidden_dimに変換
        context_all = torch.cat([context_num, context_emb], dim=0)  # [context_input_dim]
        context_out = self.context_fc(context_all.unsqueeze(0))  # [1, hidden_dim]

        # contextを各馬に複製
        context_expand = context_out.expand(horse_features.size(0), -1)  # [頭数, hidden_dim]

        # ===== contextを初期化ベクトルとして加算 =====
        # horse_featuresとcontext_expandの次元を合わせる必要あり
        if horse_features.size(1) != context_expand.size(1):
            # 線形変換で合わせる
            # horse_features = self.horse_proj(horse_features)  # [頭数, hidden_dim]
            pass

        # ====== 順位スコア正規化（オプション） ======
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)  # 高順位ほど1に近い
            norm_rank = norm_rank.unsqueeze(1)  # [頭数, 1]

            # ====== ゲーティング加算 ======
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = horse_features + gate * context_expand
        else:
            # 従来のcontext加算
            horse_features = horse_features + context_expand

        # horse_features = horse_features + context_expand  # 要素ごと加算

        # ====== 交互作用（Hadamard product） ======
        # interaction = horse_features * context_expand  # [頭数, hidden_dim]

        # # ====== 順位スコア正規化ゲート後 ======
        # combined = torch.cat([horse_features, context_expand, interaction], dim=1)

        # ===== 交互作用（3種類） =====
        mul_interaction = horse_features * context_expand
        add_interaction = horse_features + context_expand
        diff_interaction = torch.abs(horse_features - context_expand)

        # ===== まとめて結合 =====
        combined = torch.cat([
            horse_features,
            context_expand,
            mul_interaction,
            add_interaction,
            diff_interaction
        ], dim=1)

        # ====== 最終出力 ======
        out = self.fc(combined)  # [頭数, 1]

        # ---- Residual（horse_features由来の補正） ----
        resid = self.horse_residual(horse_features)   # [頭数, 1]
        out = out + resid

        # --- Residual Connection ---
        # MLPを通した特徴に、元のcombined（馬＋context）をskip接続
        # out = out + self.residual_proj(combined)

        # === 🔹 ここを追加：Softmax勝率化 ===
        # scores = out.squeeze(-1)  # [頭数]
        # probs = torch.softmax(scores, dim=0)

        # ===== 最終出力 =====
        return out.squeeze(-1)

class ListNetGRU(ListNet):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes,
                 emb_dim=16, hidden_dim=64, gru_hidden=32, num_past_runs=5, past_feat_dim=5):
        super().__init__(embedding_sizes, num_features, context_embedding_sizes, context_num_sizes,
                         emb_dim, hidden_dim)

        self.num_past_runs = num_past_runs
        self.past_feat_dim = past_feat_dim  # 過去走1件あたりの数値特徴数

        # 追加：GRU
        self.gru = nn.GRU(input_size=past_feat_dim, hidden_size=gru_hidden, batch_first=True)

        # horse_projの入力次元を修正（GRU出力をconcatするので +gru_hidden）
        self.horse_proj = nn.Linear(num_features + len(embedding_sizes) * emb_dim + gru_hidden, hidden_dim)

    def forward(self, x, cat_X, context_num, context_cat, past_runs=None, rank_scores=None):
        """
        past_runs: (頭数, num_past_runs, past_feat_dim)
        """
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        horse_features = torch.cat([x, emb], dim=1)  # [頭数, num_features + emb_dim_total]

        # ===== GRU過去走 =====
        if past_runs is not None:
            gru_out, _ = self.gru(past_runs)           # [頭数, T, gru_hidden]
            gru_last = gru_out[:, -1, :]               # 最終ステップのみ
            horse_features = torch.cat([horse_features, gru_last], dim=1)

        # ===== context処理 =====
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)
        context_num = context_num[0]
        context_all = torch.cat([context_num, context_emb], dim=0)
        context_out = self.context_fc(context_all.unsqueeze(0))
        context_expand = context_out.expand(horse_features.size(0), -1)

        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)
            norm_rank = norm_rank.unsqueeze(1)
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = self.horse_proj(horse_features)  # 次元合わせ
            horse_features = horse_features + gate * context_expand
        else:
            horse_features = self.horse_proj(horse_features)
            horse_features = horse_features + context_expand

        interaction = horse_features * context_expand
        combined = torch.cat([horse_features, context_expand, interaction], dim=1)
        out = self.fc(combined)
        out = out + self.residual_proj(combined)

        # ---- ★ 追加：logit clipping ★ ----
        out = torch.tanh(out) * 5
        return out.squeeze(-1)


# --- LambdaRank上位k損失 ---
def lambdarank_loss_at_k(preds, labels, k=3):
    n = preds.shape[0]
    diff = preds.unsqueeze(0) - preds.unsqueeze(1)
    pred_prob = torch.sigmoid(diff)
    target = (labels.unsqueeze(0) > labels.unsqueeze(1)).float()

    gain = torch.pow(2.0, labels.float()) - 1.0
    _, rank_order = torch.sort(preds, descending=True)
    rank_pos = torch.argsort(rank_order).float() + 1.0
    discount = 1.0 / torch.log2(rank_pos + 1.0)
    delta_ndcg = torch.abs((gain.unsqueeze(0) * discount.unsqueeze(0) - gain.unsqueeze(1) * discount.unsqueeze(1)))

    mask_k = (rank_pos <= k).float()
    weight_k = mask_k.unsqueeze(0) * mask_k.unsqueeze(1)
    delta_ndcg = delta_ndcg * weight_k

    mask = torch.ones_like(target, dtype=torch.bool)
    mask.fill_diagonal_(False)
    bce = F.binary_cross_entropy(pred_prob[mask], target[mask], reduction='none')
    loss = torch.mean(delta_ndcg[mask] * bce)
    return loss

def make_smooth_relevance_labels(df, max_rel=3.0):
    return (max_rel / np.log1p(df['着順'] + 1)).clip(0, max_rel)


# --- 期待値ベース損失 ---
def safe_ev_loss(preds, odds, is_win, clip_logits=10.0, clip_odds=50.0):
    preds = torch.clamp(preds, -clip_logits, clip_logits)  # 勾配爆発防止
    odds = torch.log1p(odds)
    odds = torch.clamp(odds, 1.0, clip_odds)               # オッズ極端値防止
    probs = F.softmax(preds, dim=0)
    # ev = probs * odds
    # loss = -torch.sum(is_win * ev)
    # 勝ち馬の期待値と全体の期待値の差を最小化
    ev_true = torch.sum(is_win * odds)
    ev_pred = torch.sum(probs * odds)
    loss = torch.abs(ev_true - ev_pred)  # → 確率分布全体が期待値構造を学ぶ
    if torch.isnan(loss) or torch.isinf(loss):
        # print("⚠️ EV loss unstable:", preds, odds, is_win)
        loss = torch.tensor(0.0, device=preds.device)
    return loss

def roi_policy_loss(preds, odds, is_win, temperature=1.0):
    probs = F.softmax(preds / temperature, dim=0)
    reward = is_win * odds - 1.0  # 配当 - 賭け金
    baseline = torch.sum(probs * reward)  # 平均ROI（baselineで分散低減）
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(probs + 1e-9) * advantage)
    return loss

def roi_policy_sigmoid_loss(preds, odds, is_win, temperature=1.0):
    # 各馬が買われる確率（方策）をsigmoidで定義
    buy_prob = torch.sigmoid(preds / temperature)
    reward = is_win * odds - 1.0
    baseline = torch.sum(buy_prob * reward) / buy_prob.sum().clamp(min=1.0)
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(buy_prob + 1e-9) * advantage)
    return loss

def differentiable_roi_loss(preds, odds, is_win, threshold=1.0, beta=10.0):
    probs = torch.sigmoid(preds)
    buy_prob = torch.sigmoid(beta * (probs * odds - threshold))
    roi = torch.sum(buy_prob * (is_win * odds - 1.0))
    return -roi  # ROIを最大化

def combined_roi_loss(preds, odds, is_win, alpha=0.5):
    return alpha * roi_policy_loss(preds, odds, is_win) \
         + (1 - alpha) * differentiable_roi_loss(preds, odds, is_win)

# --- combined loss ---
def combined_loss(preds, labels, odds, is_win, pairwise, weight_mode, k=3, alpha=0.05):
    # loss_rank = lambdarank_loss_at_k(preds, labels)
    # loss_ev = roi_weighted_loss(preds, odds, is_win)
    # loss_ev = weighted_roi_loss(preds, odds, is_win)
    # loss_ev = place_listnet_loss(preds, is_win)
    # loss_ev = place_ev_loss(preds, is_win, odds)
    # loss_ev = combined_place_loss(preds, is_win, odds)
    # loss_ev = place_contrastive_loss(preds, is_win)
    # loss_ev = topk_place_rank_loss(preds, is_win, k=3)
    # loss_ev = smooth_place_listnet_loss(preds, is_win)
    # loss_ev = place_contrastive_loss(preds, is_win)
    # loss_ev = calibrated_place_listnet_loss(preds, is_win)
    # loss_ev = topk_place_rank_loss_roi(preds, is_win, odds)
    # loss_ev = place_listnet_with_margin(preds, is_win)
    # loss_ev = topk_place_rank_loss_roi(preds, is_win, odds)
    # loss_ev = soft_topk_hit_loss(preds, is_win)
    # loss_ev = sharpe_roi_loss(preds, odds, is_win)
    # loss_ev = diff_roi_loss(preds, odds, is_win)
    # loss_ev = roi_with_calibration_loss(preds, odds, is_win)
    # loss_ev = pairwise_roi_loss(preds, odds, is_win)
    # loss_ev = expected_value_loss(preds, odds, is_win)
    # loss_ev = ev_huber_loss(preds, odds, is_win)
    # loss_rank = listnet_loss(preds, labels)

    # 1) EV の計算（典型）
    ev = is_win * odds - 1.0   # これは学習時の“実際の”EV実績。学習時のみ可能。

    # 2) uar_loss を計算
    uar = utility_aware_ranking_loss_roi(preds, ev, odds * is_win,
                                    margin=0, pairwise=pairwise,
                                    weight_mode=weight_mode)
    
    
    
    
    # # スケール調整
    # return uar + alpha * loss_rank
    # loss_ev = loss_ev / max(1.0, torch.abs(loss_ev))
    # return loss_ev
    # return alpha * loss_rank + (1 - alpha) * loss_ev
    return uar

    # return loss_rank + alpha * loss_ev

def utility_aware_ranking_loss_roi(
    preds,
    values,       # 勝つ確率など
    payouts,      # 配当
    mask=None,
    margin=0.0,
    pairwise='hinge',
    weight_mode='value_i',
    normalize_values=False,  # EV なら False 推奨
    eps=1e-8
):
    '''
    阪神：分割seed1, モデルseed22, fold0
    pairwise='logistic',
    weight_mode='roi'
    loss: uar単独

    中京：分割seed1, モデルseed4, fold2
    pairwise:squared_hinge
    weight_mode:ev_i
    roi:1.0983656962025317
    '''
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
        values = values.unsqueeze(0)
        payouts = payouts.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, N = preds.shape
    device = preds.device

    if mask is None:
        mask = torch.ones_like(preds)
    mask = mask.float()

    # optional normalize
    if normalize_values:
        val_sum = (values * mask).sum(dim=1)
        cnt = mask.sum(dim=1).clamp_min(1.0)

        val_mean = val_sum / cnt
        centered = values - val_mean.unsqueeze(1)

        mad = (centered.abs() * mask).sum(dim=1) / cnt
        mad = mad.clamp_min(1.0)

        values = centered / mad.unsqueeze(1)
        values = torch.clamp(values, -10.0, 10.0)

    # ペア差分
    pred_i = preds.unsqueeze(2)
    pred_j = preds.unsqueeze(1)
    diff = pred_i - pred_j

    val_i = values.unsqueeze(2)
    val_j = values.unsqueeze(1)

    payout_i = payouts.unsqueeze(2)
    payout_j = payouts.unsqueeze(1)

    mask_i = mask.unsqueeze(2)
    mask_j = mask.unsqueeze(1)
    pair_mask = mask_i * mask_j
    diag = torch.eye(N, device=device).unsqueeze(0)
    pair_mask = pair_mask * (1 - diag)

    # 重み付け
    if weight_mode == 'value_i':
        w = F.relu(val_i)

    elif weight_mode == 'roi':
        expected_i = val_i * payout_i
        expected_j = val_j * payout_j
        w = F.relu(expected_i - expected_j)

    elif weight_mode == 'abs_value_diff':
        w = (val_i - val_j).abs()

    elif weight_mode == 'ev_i':
        w = F.relu(val_i * payout_i)

    elif weight_mode == 'softmax_ev':
        ev = values * payouts
        sm = torch.softmax(ev, dim=1)
        w = sm.unsqueeze(2)

    elif weight_mode == 'rank_focus':
        K = min(3, N)
        topk = torch.topk(preds, K, dim=1).indices
        focus = torch.zeros_like(preds)
        focus.scatter_(1, topk, 1.0)
        w = focus.unsqueeze(2)

    elif weight_mode == 'focal_roi':
        ev_diff = (val_i * payout_i) - (val_j * payout_j)
        base = F.relu(ev_diff)
        focal = (1.0 - torch.sigmoid(diff)).pow(2)
        w = base * focal

    elif weight_mode == 'odds_aware':
        w = F.relu((val_i * payout_i) / (payout_i + eps))

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    # ★ 共通処理 ★
    weights = w * pair_mask

    # all-zero weights回避
    sum_w = weights.sum(dim=(1,2))
    zero_mask = (sum_w < eps)
    if zero_mask.any():
        weights[zero_mask] = pair_mask[zero_mask]

    # ペアワイズ損失
    if pairwise == 'hinge':
        pair_loss = F.relu(margin - diff)
    
    elif pairwise == 'squared_hinge':
        pair_loss = F.relu(margin - diff) ** 2

    elif pairwise == 'logistic':
        pair_loss = F.softplus(-diff)

    elif pairwise == 'bpr':
        pair_loss = -F.logsigmoid(diff)

    elif pairwise == 'exp':
        pair_loss = torch.exp(-diff)

    elif pairwise == 'soft_margin':
        pair_loss = F.softplus(margin - diff)

    elif pairwise == 'tanh':
        pair_loss = 1.0 - torch.tanh(diff)
        
    else:
        raise ValueError(f"Unknown pairwise: {pairwise}")

    weighted = pair_loss * weights
    sum_loss = weighted.sum(dim=(1,2))
    sum_weights = weights.sum(dim=(1,2)).clamp_min(eps)

    return (sum_loss / sum_weights).mean()

def ev_huber_loss(pred, odds, is_win, delta=1.0):
    target = is_win * odds - 1
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic + 0.5 * delta**2
    return torch.where(abs_diff < delta, quadratic**2 * 0.5, linear).mean()

def listnet_loss(pred_scores, true_ranks):
    """
    pred_scores: [N] モデルのスコア
    true_ranks: [N] 着順 (1〜18など)
    """

    # --- ground truth をソフト化（順位が低いほど強い → マイナス付ける） ---
    y_true = -true_ranks.float()
    P_y = torch.softmax(y_true, dim=0)

    # --- model output を softmax ---
    P_z = torch.softmax(pred_scores, dim=0)

    # cross entropy (ListNet の定義どおり)
    loss = -(P_y * torch.log(P_z + 1e-12)).sum()
    return loss

def expected_value_loss(preds, odds, is_win):
    # p = 勝率の推定（softmax か sigmoid）
    p = torch.softmax(preds, dim=0)

    # true outcome probability (one-hot)
    y = is_win

    # クロスエントロピーで勝率を学ぶ
    ce = torch.mean(-y * torch.log(p + 1e-9))

    # EV = p * odds - 1 の最大化
    ev = torch.sum(p * odds) - 1

    return ce - 0.1 * ev    # 0.05〜0.2で調整

def pairwise_roi_loss(preds, odds, is_win, margin=0.0):
    win_idx = is_win.argmax()
    win_pred = preds[win_idx]
    win_roi = odds[win_idx] - 1

    loss = 0
    cnt = 0

    for i in range(len(preds)):
        if i == win_idx:
            continue
        diff = (win_pred - preds[i])
        target_margin = max(0, win_roi - (odds[i] - 1))  # ROI差 margin
        loss += torch.relu(target_margin - diff)
        cnt += 1

    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)
    roi = torch.sum(w * (is_win * odds - 1))

    return loss/cnt - roi

def sharpe_roi_loss(preds, odds, is_win):
    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)

    reward = is_win * odds - 1      # 各馬の単勝リターン
    roi = torch.sum(w * reward)     # 期待ROI

    # 分散（リスク）
    variance = torch.sum(w * (reward - roi)**2)

    # シャープレシオを最大化
    sharpe = roi / torch.sqrt(variance + 1e-9)

    return -sharpe

def roi_with_calibration_loss(preds, odds, is_win, alpha=0.2):
    '''
    中山：分割seed1, モデルseed1, fold1, alpha=0.5
    '''
    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)

    reward = is_win * odds - 1
    roi = torch.sum(w * reward)

    # 予測勝率（softmax）
    probs = torch.softmax(preds, dim=0)

    # Brier（確率校正）
    brier = torch.mean((probs - is_win)**2)

    return -(roi - alpha * brier)

def roi_weighted_loss(preds, odds, is_win):
    '''
    東京：分割・モデルともにseed1で使用 fold0
    '''
    weights = torch.relu(preds) / torch.sum(torch.relu(preds) + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(weights * reward)
    return -roi  # ROIを直接最大化

def weighted_roi_loss(preds, odds, is_win):
    # 低オッズの馬に重みを付ける
    # 例えば 〜7倍帯は w=2、それ以上は w=1
    weights_odds = torch.where(odds < 7, 2.0, 1.0)
    
    weights = torch.relu(preds) / (torch.sum(torch.relu(preds)) + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(weights * reward * weights_odds)
    return -roi

def place_listnet_loss(logits, is_in, top_k=3):
    # 可能性あり
    """
    logits: モデル出力
    is_in: 0/1
    """
    # 複勝的中を softmax 確率に変換
    y_true = is_in.float()
    y_true = y_true / y_true.sum()  # 0除算注意
    y_pred = torch.softmax(logits, dim=0)
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-8))
    return loss

def place_ev_loss(logits, is_in, place_odds):
    # 可能性あり
    """
    logits: モデル出力（sigmoidで確率に変換）
    is_in: 0/1
    place_odds: 当たりはオッズ、外れは0
    """
    place_odds_fixed = place_odds.clone()
    place_odds_fixed[place_odds_fixed==0] = 1.0
    prob = torch.sigmoid(logits)
    reward = is_in * (place_odds_fixed - 1) - (1 - is_in)
    ev = prob * reward

    return -ev.mean()

def combined_place_loss(logits, is_in, place_odds, alpha=0, top_k=3):
    """
    alpha: ListNet loss の重み（0〜1）
    1-alpha が EV loss の重み
    """
    # ListNet的損失
    listnet_loss = place_listnet_loss(logits, is_in, top_k=top_k)

    # EV損失
    ev_loss = place_ev_loss(logits, is_in, place_odds)

    # 合成
    loss = alpha * listnet_loss + (1 - alpha) * ev_loss
    return loss

def place_contrastive_loss(logits, is_in, temperature=0.07):
    pos = logits[is_in == 1]
    neg = logits[is_in == 0]

    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)

    loss_list = []

    for p in pos:
        # 正例と負例をまとめてスケーリング
        scores = torch.cat([p.unsqueeze(0), neg]) / temperature

        # logsumexp による安定化
        lse = torch.logsumexp(scores, dim=0)

        # numerator の log = p/temperature
        log_num = p / temperature

        loss_list.append(-(log_num - lse))

    return torch.mean(torch.stack(loss_list))

def topk_place_rank_loss(logits, is_in, k=3, margin=1.0):
    '''
    [top評価結果test ブートストラップ評価] レース数: 823 的中率: 28.81%（95%CI: 25.76% ～ 31.83%） 回収率: 85.41%（95%CI: 65.46% ～ 110.26%）
    '''
    pos = logits[is_in == 1]
    neg = logits[is_in == 0]

    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)

    # 上位Kに入ってほしい正例
    top_neg = torch.topk(neg, k=min(k, len(neg))).values

    loss = 0
    count = 0
    for p in pos:
        for n in top_neg:
            loss += torch.relu(margin - (p - n))
            count += 1

    return loss / count

def smooth_place_listnet_loss(logits, is_in, eps=0.1):
    '''
    [ex評価結果test ブートストラップ評価] レース数: 1748 的中率: 59.96%（95%CI: 57.67% ～ 62.24%） 回収率: 90.69%（95%CI: 86.14% ～ 95.34%）
    '''
    k = is_in.sum()

    if k == 0:
        return torch.tensor(0.0, device=logits.device)

    # 正例: (1-eps)/k、負例: eps/(N-k)
    N = len(logits)
    y_true = torch.zeros(N, device=logits.device)
    y_true[is_in == 1] = (1 - eps) / k
    y_true[is_in == 0] = eps / (N - k)

    y_pred = torch.softmax(logits, dim=0)

    loss = -torch.sum(y_true * torch.log(y_pred + 1e-8))
    return loss

def topk_place_rank_loss_roi(
    logits, is_in, payout, bet=None, k=1,
    margin=1.0, alpha=0.1, eps=1e-9
):
    """
    ROI 寄与度を調整できる top-k複勝 RankLoss
    """
    if bet is None:
        bet = torch.ones_like(payout) * 100

    roi = payout / (bet + eps)

    # ROI → log1p → α乗で寄与度調整
    roi_weight = torch.log1p(roi) ** alpha

    pos_idx = (is_in == 1)
    pos = logits[pos_idx]
    pos_w = roi_weight[pos_idx]

    neg = logits[~pos_idx]
    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)

    top_neg = torch.topk(neg, k=min(k, len(neg))).values

    loss = 0
    count = 0

    for p, w in zip(pos, pos_w):
        for n in top_neg:
            raw = torch.relu(margin - (p - n))
            loss += w * raw
            count += 1

    return loss / (count + eps)

def soft_topk_hit_loss(logits, is_in, k=3, tau=1.0):
    # softmax で順位確率
    probs = torch.softmax(logits / tau, dim=0)

    # 正例の確率合計が大きくなるように
    hit_prob = probs[is_in == 1].sum()

    # 最大化したい → マイナスにして minimize
    return -hit_prob



def roi_adjusted_score(preds, odds, alpha=0.1):
    """
    preds: [num_horses] モデル出力スコア
    odds: [num_horses] 単勝オッズ
    alpha: ROI補正の強さ
    """
    # ログオッズで補正（勾配に影響させない）
    roi_factor = torch.log1p(odds)
    adjusted = preds * (1 + alpha * roi_factor)
    
    return adjusted




def make_rank_labels(df, group_col='レースID', pos_col='着順', n_bins=18):
    """
    出走頭数を使って順位を0..n_binsに段階化する。
    rel_raw = (出走頭数 - 着順 + 1) / 出走頭数  を使い、0..1を等幅n_binsに分割。
    returns: df with new column 'rank_label' (int 0..n_bins)
    """
    df = df.copy()
    rel_series = pd.Series(index=df.index, dtype=float)

    for race_id, g in df.groupby(group_col):
        n = len(g)
        # 相対スコア: 1.0 (1着) ... 1/n (最下位)
        rel_raw = (n - g[pos_col].values + 1) / n
        rel_series.loc[g.index] = rel_raw

    # 0..1 を n_bins 等分して離散化（0..n_bins）
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize だと右端clampなので -1 調整
    labels = np.digitize(rel_series.values, bins, right=True) - 1
    labels = np.clip(labels, 0, n_bins).astype(int)

    df['rank_label'] = labels
    return df

def make_label_gain(n_bins=18, mode='sqrt'):
    """
    n_bins: label の最大値
    mode: 'linear', 'sqrt', 'exp', 'dcg' などで調整
    returns: list length n_bins+1
    """
    if mode == 'linear':
        return list(range(0, n_bins+1))
    if mode == 'sqrt':
        return [0] + [np.sqrt(i) for i in range(1, n_bins+1)]
    if mode == 'exp':
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    if mode == 'dcg':  # DCG風 (2^rel-1)
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    # default linear
    return list(range(0, n_bins+1))


def labmdarank_lgb(train_df, val_df, test_df, feature_cols, target_col, embedding_cols, fold):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # パラメータ設定
    rate = 0.01
    seed=42
    gain_list = make_label_gain()
    group_train = train_df.groupby("レースID").size().to_list()
    group_val = val_df.groupby("レースID").size().to_list()

    lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], categorical_feature=embedding_cols, group=group_train)
    lgb_eval = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], categorical_feature=embedding_cols, reference=lgb_train, group=group_val)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': rate,
        'random_state': seed,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'num_boost_round': 10000
    }
    ####################################################################################

    # '''
    # クロスバリデーションによるハイパーパラメータの探索 3fold
    tuner = lgb.LightGBMTunerCV(params,
                                lgb_train,
                                folds=GroupKFold(n_splits=3),
                                # categorical_feature = cat_list,
                                return_cvbooster=True,
                                verbose_eval=False
                                )

    # # ハイパーパラメータ探索の実行
    tuner.run()

    # # サーチしたパラメータの表示
    best_params = tuner.best_params
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    print(tuner.best_score)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': rate,
        'random_state': seed,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'num_boost_round': 10000,
        'feature_pre_filter': best_params['feature_pre_filter'],
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq':  best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples'],
    }

    evals_result = {}
    model = lgbm.train(params,
                    lgb_train,  # トレーニングデータの指定
                    valid_names=['valid', 'train'],    # 学習経過で表示する名称
                    valid_sets=[lgb_eval, lgb_train],  # 先頭が early stopping 判定対象
                    # categorical_feature = cat_list,
                    callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),
                                lgbm.record_evaluation(evals_result)]
                    )

    # pklファイルとしてモデルを保存
    with open(f"./model/tokyo_lambdarank_{fold}.pickle", "wb") as mk:
        pickle.dump(model, mk)

    # テストデータの予測 (予測クラスを返す)
    val_df['pred_score'] = model.predict(val_df[feature_cols], num_iteration=model.best_iteration)
    test_df['pred_score'] = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)

    val_df.to_csv(f'./csv/tokyo_result_lambdarank_val_{fold}.csv', index=False)
    test_df.to_csv(f'./csv/tokyo_result_lambdarank_test_{fold}.csv', index=False)



# def listnet_loss(preds, labels, gain):
#     preds = preds - preds.max()
#     Pz = torch.softmax(preds, dim=0)

#     # 勝率に gain を掛けて重み付け
#     weighted_labels = labels * gain

#     loss = -torch.sum(weighted_labels * torch.log(Pz + 1e-12))
#     return loss


def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()
    
    # with open('./model/platt.pkl', 'rb') as f:
    #     platt = pickle.load(f)

    # val_df['pred_score'] = platt.predict_proba(np.array(val_df['pred_score']).reshape(-1, 1))[:, 1]

    def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))  # 安定化のために最大値を引く
            return e_x / e_x.sum()
        return softmax
    # df_sorted = val_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df_sorted[['レースID', 'pred_score']].head(30))
    
    # val_df['log_odds'] = np.log(val_df['オッズ'] + 1)
    for i in range(1, 21):
        # T = 0.1 * i  # 例：温度を0.5に設定（小さいほど尖る）
        T = 0.2
        softmax_T = make_softmax_with_temperature(T)

        val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
        val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
        # val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
        top_by_race = val_df.groupby('レースID').apply(
            lambda df: df.sort_values('expected_value', ascending=False)
        )

    #     print(top_by_race[['softmax_score', 'オッズ', 'expected_value', 'win_prob']].head(100))

        # a = 0.3
        # b = 0.7
        # val_df['expected_value'] = (val_df['softmax_score'] ** a) * (val_df['オッズ'] ** b)

        # val_df['expected_value'] = val_df['pred_score'] * val_df['log_odds']

        # 各レースごとに期待値上位3頭を取得
        # top3_ev = (
        #     val_df.sort_values(['レースID', 'pred_score'], ascending=[True, False])
        #         .groupby('レースID')
        #         .head(3)
        # )

        # # その中で pred_score 最大の馬を1頭だけ抽出
        # selected = (
        #     top3_ev.loc[top3_ev.groupby('レースID')['expected_value'].idxmax()]
        # )
        
        selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]
        selected = selected[selected['expected_value'] < 4]
        selected = selected[selected['オッズ'] > i]
        print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])
        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()
        hit_count = (selected['着順'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[評価結果 - Fold {i}]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
        pass
    # print(selected[['softmax_score', 'log_odds', 'expected_value']].sort_values('expected_value', ascending=False).head(20))

    
    # top = top[top['expected_value'] > 100]

    # for i in range(1, 21):
    #     T = 0.2
    #     softmax_T = make_softmax_with_temperature(T)

    #     val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
    #     val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
    #     top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
    #     top = top[top['expected_value'] > 0.1 * i]
    #     total_bet = len(top) * 100
    #     total_return = top['単勝オッズ'].sum()
    #     hit_count = (top['着順'] == 1).sum()
    #     roi = total_return / total_bet

    #     print(f"\n[top評価結果{i}]")
    #     print(f"レース数: {len(top)}")
    #     print(f"的中数: {int(hit_count)}")
    #     print(f"的中率: {hit_count / len(top):.2%}")
    #     print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

    return val_df

# テスト
# val_df = pd.read_csv('./csv/tokyo_result_listnet_0.csv')  # または val データ専用ファイルを読み込む

# val_df_result = evaluate_model_on_val_df(val_df, model_path='./model/tokyo_listnet_0.pth', fold=0)

# def target_encording(df, column, target):
#     tem = pd.DataFrame()
#     df_tem = pd.DataFrame()
#     df_ind = pd.DataFrame()
#     dfs = [df.iloc[i:i+int(len(df.index)/5)+1, :] for i in range(0, len(df.index), int(len(df.index) / 5) + 1)]

#     for i in range(5):
#         df_tem = dfs[i].copy()
#         df_ind = dfs.copy()
#         del df_ind[i]
#         df_ind = pd.concat([dfs[0], dfs[1], dfs[2], dfs[3]], axis=0)
#         d = df_ind.groupby(column)[target].mean()
#         dict = d.to_dict()
#         df_tem[column] = pd.to_numeric(df_tem[column].map(dict), errors='coerce')
#         tem = pd.concat([tem, df_tem], axis=0)
    
#     return tem

# def target_encoding(df, col, target, n_splits=5, random_state=42):
#     df = df.copy()
#     df[col + "_te"] = np.nan
    
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
#     for train_idx, val_idx in kf.split(df):
#         train_data = df.iloc[train_idx]
#         mapping = train_data.groupby(col)[target].mean().to_dict()
        
#         # val にマッピング。未知カテゴリは -1 で埋める
#         val_values = df.iloc[val_idx][col]  # ← loc ではなく iloc
#         df.iloc[val_idx, df.columns.get_loc(col + "_te")] = val_values.map(mapping).fillna(-1)
    
#     full_mapping = df.groupby(col)[target].mean().to_dict()
#     return df, full_mapping

def target_encoding(df, col, target, n_splits=5, alpha=20, random_state=42):
    df = df.copy()
    df[col + "_te"] = np.nan

    global_mean = df[target].mean()  # 全体平均を事前確率として使う
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(df):
        train_data = df.iloc[train_idx]

        # カテゴリごとの count と mean を計算
        stats = train_data.groupby(col)[target].agg(['mean', 'count'])

        # 平滑化
        stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)

        mapping = stats['smooth'].to_dict()

        val_values = df.iloc[val_idx][col]
        df.iloc[val_idx, df.columns.get_loc(col + "_te")] = val_values.map(mapping).fillna(global_mean)

    # 学習全体で最終マッピング（推論用）
    full_stats = df.groupby(col)[target].agg(['mean', 'count'])
    full_stats['smooth'] = (full_stats['mean'] * full_stats['count'] + alpha * global_mean) / (full_stats['count'] + alpha)
    full_mapping = full_stats['smooth'].to_dict()

    return df, full_mapping

def group_by_race(df_part):
        X_groups, y_groups, win_groups, payout_groups = [], [], [], []
        cat_groups, context_num_groups, context_cat_groups = [], [], []
        win_index_groups = []

        for _, g in df_part.groupby(group_col):
            X = g[feature_cols].values.astype(np.float32)
            y = g[target_col].values.astype(np.float32)  # 予測対象（勝率など）

            # 勝敗ラベルと払戻（gain用）
            is_win = g["is_win"].values.astype(np.float32)          # 0 or 1
            payout = g["オッズ"].values.astype(np.float32) - 1.0    # 払戻倍率-1（gain）

            cat_X = g[embedding_cols].values.astype(np.int64)

            context_num = g[context_num_cols].iloc[0].values.astype(np.float32)
            context_cat = g[context_cat_cols].iloc[0].values.astype(np.int64)

            num_horses = len(g)
            context_num = np.tile(context_num, (num_horses, 1))
            context_cat = np.tile(context_cat, (num_horses, 1))

            # --- torch変換 ---
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            win_t = torch.tensor(is_win, dtype=torch.float32)
            payout_t = torch.tensor(payout, dtype=torch.float32)
            cat_t = torch.tensor(cat_X, dtype=torch.long)
            context_num_t = torch.tensor(context_num, dtype=torch.float32)
            context_cat_t = torch.tensor(context_cat, dtype=torch.long)

            # --- 勝者index（CrossEntropy用）---
            if win_t.sum() == 1:
                win_idx = torch.argmax(win_t).long()
            else:
                win_idx = torch.tensor(-1).long()  # 異常ケース（全頭負など）
            win_index_groups.append(win_idx)

            X_groups.append(X_t)
            y_groups.append(y_t)
            win_groups.append(win_t)
            payout_groups.append(payout_t)
            cat_groups.append(cat_t)
            context_num_groups.append(context_num_t)
            context_cat_groups.append(context_cat_t)

        return (
            X_groups, y_groups, cat_groups,
            context_num_groups, context_cat_groups,
            win_groups, payout_groups, win_index_groups
        )

def group_by_race_fuku(df_part):
    X_groups, y_groups, win_groups, payout_groups = [], [], [], []
    cat_groups, context_num_groups, context_cat_groups = [], [], []
    win_index_groups = []

    for _, g in df_part.groupby(group_col):
        X = g[feature_cols].values.astype(np.float32)
        y = g[target_col].values.astype(np.float32)  # 予測対象（勝率など）

        # 勝敗ラベルと払戻（gain用）
        is_win = g["複勝_hit"].values.astype(np.float32)          # 0 or 1
        payout = (g["複勝払戻"].values.astype(np.float32) / 100)    # 払戻倍率-1（gain）

        cat_X = g[embedding_cols].values.astype(np.int64)

        context_num = g[context_num_cols].iloc[0].values.astype(np.float32)
        context_cat = g[context_cat_cols].iloc[0].values.astype(np.int64)

        num_horses = len(g)
        context_num = np.tile(context_num, (num_horses, 1))
        context_cat = np.tile(context_cat, (num_horses, 1))

        # --- torch変換 ---
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        win_t = torch.tensor(is_win, dtype=torch.float32)
        payout_t = torch.tensor(payout, dtype=torch.float32)
        cat_t = torch.tensor(cat_X, dtype=torch.long)
        context_num_t = torch.tensor(context_num, dtype=torch.float32)
        context_cat_t = torch.tensor(context_cat, dtype=torch.long)

        # --- 勝者index（CrossEntropy用）---
        if win_t.sum() == 1:
            win_idx = torch.argmax(win_t).long()
        else:
            win_idx = torch.tensor(-1).long()  # 異常ケース（全頭負など）
        win_index_groups.append(win_idx)

        X_groups.append(X_t)
        y_groups.append(y_t)
        win_groups.append(win_t)
        payout_groups.append(payout_t)
        cat_groups.append(cat_t)
        context_num_groups.append(context_num_t)
        context_cat_groups.append(context_cat_t)

    return (
        X_groups, y_groups, cat_groups,
        context_num_groups, context_cat_groups,
        win_groups, payout_groups, win_index_groups
    )

# def group_by_race_fuku(df_part, past_cols):
#     """
#     df_part: DataFrame
#     past_cols: list of list, 過去走の数値カラム名 [[1過去走の特徴], [2過去走の特徴], ...]
#     """
#     X_groups, y_groups, win_groups, payout_groups = [], [], [], []
#     cat_groups, context_num_groups, context_cat_groups = [], [], []
#     win_index_groups = []
#     past_runs_groups = []

#     for _, g in df_part.groupby(group_col):
#         X = g[feature_cols].values.astype(np.float32)
#         y = g[target_col].values.astype(np.float32)

#         is_win = g["複勝_hit"].values.astype(np.float32)
#         payout = (g["複勝払戻"].values.astype(np.float32) / 100)

#         cat_X = g[embedding_cols].values.astype(np.int64)

#         context_num = g[context_num_cols].iloc[0].values.astype(np.float32)
#         context_cat = g[context_cat_cols].iloc[0].values.astype(np.int64)

#         num_horses = len(g)
#         context_num = np.tile(context_num, (num_horses, 1))
#         context_cat = np.tile(context_cat, (num_horses, 1))

#         # ===== GRU用過去走特徴 =====
#         past_features = g[add_cols].values.astype(np.float32)  # shape=(num_horses, num_past_runs * feat_dim)
#         past_feat_dim = len(past_cols[0])
#         num_past_runs = len(past_cols)
#         past_features = past_features.reshape(num_horses, num_past_runs, past_feat_dim)
#         past_runs_t = torch.tensor(past_features, dtype=torch.float32)

#         # ===== Torch変換 =====
#         X_t = torch.tensor(X, dtype=torch.float32)
#         y_t = torch.tensor(y, dtype=torch.float32)
#         win_t = torch.tensor(is_win, dtype=torch.float32)
#         payout_t = torch.tensor(payout, dtype=torch.float32)
#         cat_t = torch.tensor(cat_X, dtype=torch.long)
#         context_num_t = torch.tensor(context_num, dtype=torch.float32)
#         context_cat_t = torch.tensor(context_cat, dtype=torch.long)

#         # 勝者index（CrossEntropy用）
#         if win_t.sum() == 1:
#             win_idx = torch.argmax(win_t).long()
#         else:
#             win_idx = torch.tensor(-1).long()

#         X_groups.append(X_t)
#         y_groups.append(y_t)
#         win_groups.append(win_t)
#         payout_groups.append(payout_t)
#         cat_groups.append(cat_t)
#         context_num_groups.append(context_num_t)
#         context_cat_groups.append(context_cat_t)
#         win_index_groups.append(win_idx)
#         past_runs_groups.append(past_runs_t)

#     return (
#         X_groups, y_groups, cat_groups,
#         context_num_groups, context_cat_groups,
#         win_groups, payout_groups, win_index_groups,
#         past_runs_groups  # 新規追加
#     )

def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)


# 各レースごとに計算して平均をとる
def calc_mean_ndcg(df, label_col='着順', score_col='pred_score', k=3):
    ndcgs = []

    for _, group in df.groupby("レースID"):
        # relevance: 小さい着順ほど重要なので逆にする
        # 例: 1着=3pt, 2着=2pt, ...（この例は出走頭数3の場合）
        max_rank = group[label_col].max()
        relevance = max_rank - group[label_col] + 1
        
        # sklearnは shape=(1, n_samples) の形式を求める
        true_relevance = [relevance.values]
        pred_scores = [group[score_col].values]
        
        # NDCG@k で計算
        score = ndcg_score(true_relevance, pred_scores, k=k)
        ndcgs.append(score)

    return np.mean(ndcgs)

def embedding_init():
    # emb = feature_category + diff_category_place + diff_category_field
    emb = feature_category
    return emb

def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)     

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader の worker 初期化用
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

# def time_series_group_cv_3split(df, group_col="レースID", n_splits=5):
#     """
#     時系列順（レースID昇順）に基づくリーク防止付きクロスバリデーション
#     各foldで train / val / test の3分割を生成する
#     """
#     unique_races = np.sort(df[group_col].unique())
#     n_races = len(unique_races)
#     fold_size = n_races // (n_splits + 2)  # test分も含めて少し余裕をもたせる

#     splits = []

#     for i in range(n_splits):
#         # 各foldで範囲を決める
#         train_end = (i + 1) * fold_size
#         val_end = train_end + fold_size
#         test_end = val_end + fold_size

#         if test_end > n_races:
#             break  # データが足りなくなったら終了

#         train_races = unique_races[:train_end]
#         val_races = unique_races[train_end:val_end]
#         test_races = unique_races[val_end:test_end]

#         train_idx = df[df[group_col].isin(train_races)].index
#         val_idx = df[df[group_col].isin(val_races)].index
#         test_idx = df[df[group_col].isin(test_races)].index

#         splits.append((train_idx, val_idx, test_idx))

#     return splits

# def time_series_group_cv_3split_2025(df, group_col="レースID", n_splits=5):
#     """
#     時系列順（レースID昇順）に基づくリーク防止付きクロスバリデーション
#     各foldで train / val / test の3分割を生成
#     2025年のデータは別 df として返す
#     """
#     # 2025年データを切り離す
#     # 2025年と2024年データを切り離す
#     df_2025 = df[df[group_col].astype(str).str[:4].isin(["2025", "2024"])].reset_index(drop=True)
#     # df_2025 = df[df[group_col].astype(str).str[:4] == "2025"].reset_index(drop=True)
#     # df_rest = df[df[group_col].astype(str).str[:4] != "2025"].reset_index(drop=True)
#     df_rest = df[~df[group_col].astype(str).str[:4].isin(["2024", "2025"])].reset_index(drop=True)

#     unique_races = np.sort(df_rest[group_col].unique())
#     n_races = len(unique_races)
#     fold_size = n_races // (n_splits + 2)  # train/val/test 含む

#     splits = []
#     train_start = 0

#     for i in range(n_splits):
#         train_end = (i + 1) * fold_size
#         val_end = train_end + fold_size
#         test_end = val_end + fold_size

#         if test_end > n_races:
#             break  # データが足りなければ終了

#         train_races = unique_races[train_start:train_end]
#         val_races = unique_races[train_end:val_end]
#         test_races = unique_races[val_end:test_end]

#         train_idx = df_rest[df_rest[group_col].isin(train_races)].index
#         val_idx = df_rest[df_rest[group_col].isin(val_races)].index
#         test_idx = df_rest[df_rest[group_col].isin(test_races)].index

#         splits.append((train_idx, val_idx, test_idx))

#         train_start = train_end

#     return splits, df_2025, df_rest

def time_series_group_cv_3split_2025(df, group_col="レースID", n_folds=5):
    """
    時系列順を保ちつつ、ランダムなtest比率（0.2〜0.3）でfoldを繰り返すCV。
    各foldで train < val < test の順序を維持。
    2024年以降は df_post2024 に切り離して返す。
    スライドは行わず、各foldの分割点を乱数で決定。
    """

    # --- ① 2024年以降を分離 ---
    # df_post2024 = df[df[group_col].astype(str).str[:4].isin(["2025"])].reset_index(drop=True)
    # df_rest = df[~df[group_col].astype(str).str[:4].isin(["2025"])].reset_index(drop=True)
    df_rest = df

    # --- ② レースIDでソート ---
    unique_races = np.sort(df_rest[group_col].unique())
    n_races = len(unique_races)
    stage_splits = []

    for i in range(n_folds):
        test_ratio = random.uniform(0.1, 0.3)
        val_ratio = 0.1
        train_ratio = 1 - (test_ratio + val_ratio)

        # --- ③ 分割点（比率に基づいて計算）
        train_end = int(n_races * train_ratio)
        val_end = int(n_races * (train_ratio + val_ratio))

        train_races = unique_races[:train_end]
        val_races = unique_races[train_end:val_end]
        test_races = unique_races[val_end:]

        train_idx = df_rest[df_rest[group_col].isin(train_races)].index
        val_idx = df_rest[df_rest[group_col].isin(val_races)].index
        test_idx = df_rest[df_rest[group_col].isin(test_races)].index

        stage_splits.append((train_idx, val_idx, test_idx))

    # return stage_splits, df_post2024, df_rest
    return stage_splits, df_rest

def time_series_train_val_split(df, group_col="レースID", train_ratio=0.8):
    """
    時系列順（レースID昇順）でリークを防ぎながら、
    train : val = 8 : 2 に分割し、test は val のダミーを返す
    """
    unique_races = np.sort(df[group_col].unique())
    n_races = len(unique_races)

    # train / val の境界
    train_end = int(n_races * train_ratio)

    train_races = unique_races[:train_end]
    val_races = unique_races[train_end:]

    # データ分割
    train_df = df[df[group_col].isin(train_races)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_races)].reset_index(drop=True)
    test_df = val_df.copy()  # ダミー

    return train_df, val_df, test_df

def generate_stat_features_by_course_train(
    df,
    candidate_cols,
    date_col="レースID",
    target_col="着順",
    max_comb=2,
    n_sample=None,
    add_course_distance=True,
    smooth_prior=10,
    verbose=True,
    save_mapping_path="./pickle-dict/winrate_mapping_by_course.pkl",
    combo_list=None
):
    feature_list = []
    """
    【train用】コース距離ごとの統計特徴量を作成し、マッピング辞書として保存する
    ※時系列依存なし（全データで集計）
    """
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    # --- コース距離列作成 ---
    if add_course_distance and ("フィールド" in df.columns and "距離" in df.columns):
        df["コース距離"] = df["フィールド"].astype(str) + "_" + df["距離"].astype(str)

    # --- 勝率フラグ ---
    df["is_win"] = (df[target_col] == 1).astype(int)
    df["is_ren"] = (df[target_col] <= 2).astype(int)
    df["is_fuku"] = (df[target_col] <= 3).astype(int)

    # --- 組み合わせ生成 ---
    if combo_list is None:
        combo_list = []
        for r in range(1, max_comb + 1):
            combo_list += list(itertools.combinations(candidate_cols, r))
        if n_sample:
            combo_list = random.sample(combo_list, min(n_sample, len(combo_list)))

    # combo_list = [('1場所', '1距離'), ('1馬場', '1コーナー通過順'), ('人気', '1人気'), ('性', '1コーナー通過順'), ('性', '1距離'), ('父馬', '1距離'), ('間隔', '性'), ('騎手',), ('騎手', '1距離'), ('齢',)]

    # --- マッピング辞書 ---
    winrate_mapping = {}

    # --- コース距離ごとに処理 ---
    for course, df_course in tqdm(df.groupby("コース距離"), disable=not verbose, desc="コース距離単位処理"):
        df_course = df_course.copy()

        for cols in combo_list:
            name_base = "_".join(cols)

            # グループごとに統計を計算（累積なし・全体平均）
            # print(cols)
            grouped = (
                df_course
                .groupby(list(cols), as_index=False)
                .agg(
                    win_sum=('is_win', 'sum'),
                    ren_sum=('is_ren', 'sum'),
                    fuku_sum=('is_fuku', 'sum'),
                    total=('is_win', 'count'),
                    avg_rank=(target_col, 'mean')
                )
            )

            # スムージング（ベイズ平滑化）
            prior_win = df_course["is_win"].mean()
            prior_ren = df_course["is_ren"].mean()
            prior_fuku = df_course["is_fuku"].mean()

            grouped[f"{name_base}_勝率"] = (grouped["win_sum"] + prior_win * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_連対率"] = (grouped["ren_sum"] + prior_ren * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_複勝率"] = (grouped["fuku_sum"] + prior_fuku * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_平均着順"] = grouped["avg_rank"]
            grouped[f"{name_base}_件数"] = grouped["total"]

            # 辞書として保存
            winrate_mapping[name_base] = winrate_mapping.get(name_base, {})
            for _, row in grouped.iterrows():
                key_val = tuple(row[c] for c in cols)
                winrate_mapping[name_base][(course, key_val)] = {
                    "勝率": row[f"{name_base}_勝率"],
                    "連対率": row[f"{name_base}_連対率"],
                    "複勝率": row[f"{name_base}_複勝率"],
                    "平均着順": row[f"{name_base}_平均着順"],
                    "件数": row[f"{name_base}_件数"]
                }

    # --- 保存 ---
    with open(save_mapping_path, "wb") as f:
        pickle.dump(winrate_mapping, f)

    print(f"✅ コース距離単位の統計特徴量生成完了。マッピングを {save_mapping_path} に保存しました。")
    return winrate_mapping, combo_list, feature_list
    
def apply_stats_from_mapping(df, stats_dict, add_course_distance=True):
    """
    train側で作成した辞書を用いて、test/valデータに統計特徴量を適用する。

    Parameters
    ----------
    df : pd.DataFrame
        適用対象データ
    stats_dict : dict
        コース距離→特徴→値→統計 の辞書
    course_col : str
        コース名のカラム
    dist_col : str
        距離のカラム
    add_course_distance : bool
        コース距離列を自動追加するか

    Returns
    -------
    df_out : pd.DataFrame
        特徴量を付与したDataFrame
    """

    df = df.copy()
    feature_list = []

    # --- コース距離列の追加（なければ）
    if add_course_distance and ("フィールド" in df.columns and "距離" in df.columns):
        if "コース距離" not in df.columns:
            df["コース距離"] = df["フィールド"].astype(str) + "_" + df["距離"].astype(str)

    # --- 対象特徴を辞書から自動抽出
    all_features = set()
    # stats_dict のトップキーを全て取得 → name_base に対応
    all_features = sorted(list(stats_dict.keys()))

    print(f"適用対象特徴: {all_features}")

    # --- 関数化（辞書アクセス安全版）
    def get_stat(course_dist, feature, value, stat_key):
        try:
            key_val = tuple(value) if isinstance(value, (list, tuple)) else value
            return stats_dict[str(feature)][(str(course_dist), key_val)][stat_key]
        except KeyError:
            return np.nan

    # --- 各特徴×統計を埋める
    for feature in all_features:
        for stat_key, jp in zip(["win_rate", "place_rate", "show_rate", "n"], ["勝率", "連対率", "複勝率", "平均着順"]):
            col_name = f"{feature}_{jp}"
            feature_list.append(col_name)
            df[col_name] = df.apply(
                lambda r: get_stat(
                    r["コース距離"],
                    feature,
                    tuple(r[c] for c in feature.split('_')),  # ここでスカラー値をタプル化
                    jp
                ),
                axis=1
            )

    return df, feature_list

def select_similar_train_races(
    df,
    group_col="レースID",
    feature_cols=None,
    target_year=2023,
    top_k=3,
    verbose=True,
):
    """
    target_year（例:2024）のレース分布に最も近い期間（レース群）を自動抽出する関数。

    Parameters
    ----------
    df : pd.DataFrame
        入力データ（レースID列を含む）
    group_col : str
        レースID列（先頭4桁が年を表す）
    feature_cols : list
        比較する特徴量のリスト（数値 or カテゴリ）
    target_year : int
        比較対象とする基準年（例:2024）
    top_k : int
        KL距離が小さい上位Nグループを学習データとして選ぶ
    verbose : bool
        tqdm表示

    Returns
    -------
    selected_df : pd.DataFrame
        KL距離が小さい上位期間を結合したtrainデータ
    kl_df : pd.DataFrame
        各期間ごとのKL距離情報
    """

    df = df.copy()
    df["year"] = df[group_col].astype(str).str[:4].astype(int)

    # --- 基準年データ ---
    base_df = df[df["year"] == target_year]
    if base_df.empty:
        raise ValueError(f"{target_year}年のレースデータが存在しません。")

    if feature_cols is None:
        # 数値またはカテゴリ特徴のみ選択
        feature_cols = [
            c for c in df.columns
            if c not in [group_col, "year"] and df[c].dtype != "object"
        ]

    kl_list = []

    # --- 年単位または月単位の比較を想定（ここでは年単位）---
    for yr, df_year in tqdm(df.groupby("year"), disable=not verbose):
        if yr == target_year:
            continue  # 基準年はスキップ

        total_kl = 0
        valid_feats = 0

        for feat in feature_cols:
            base_vals = base_df[feat].dropna()
            grp_vals = df_year[feat].dropna()

            if base_vals.empty or grp_vals.empty:
                continue

            # 数値特徴量 → ヒストグラム比較
            if np.issubdtype(base_vals.dtype, np.number):
                bins = np.histogram_bin_edges(np.concatenate([base_vals, grp_vals]), bins=20)
                p = np.histogram(base_vals, bins=bins, density=True)[0] + 1e-9
                q = np.histogram(grp_vals, bins=bins, density=True)[0] + 1e-9
            else:
                # カテゴリ特徴量
                base_probs = base_vals.value_counts(normalize=True)
                grp_probs = grp_vals.value_counts(normalize=True)
                all_cats = set(base_probs.index) | set(grp_probs.index)
                p = np.array([base_probs.get(c, 1e-9) for c in all_cats])
                q = np.array([grp_probs.get(c, 1e-9) for c in all_cats])

            kl = entropy(p, q)
            total_kl += kl
            valid_feats += 1

        if valid_feats > 0:
            kl_avg = total_kl / valid_feats
            kl_list.append({"year": yr, "kl_div": kl_avg})

    kl_df = pd.DataFrame(kl_list).sort_values("kl_div", ascending=True)

    # --- KL距離が近い上位N年のデータを抽出 ---
    top_years = kl_df.head(top_k)["year"].tolist()
    selected_df = df[df["year"].isin(top_years)].drop(columns=["year"])

    if verbose:
        print(f"\n✅ {target_year}年に最も近い分布を持つ上位{top_k}年:")
        for y, d in zip(top_years, kl_df.head(top_k)["kl_div"]):
            print(f"  - {y}: KL={d:.4f}")

    return selected_df, kl_df

def select_similar_train_races(
    df,
    group_col="レースID",
    feature_cols=None,
    target_year=2024,
    distance_type="kl",  # "kl" / "js" / "wasserstein"
    threshold=0.1,
    verbose=True,
):
    """
    target_yearのレース分布に最も近い「年×開催」を抽出する関数。
    
    distance_type:
        - "kl" : KL距離
        - "js" : Jensen-Shannon距離
        - "wasserstein" : Wasserstein距離
    threshold:
        選択する距離の閾値（KL/JSなら小さいほど近い、Wassersteinも同様）
    """
    df = df.copy()
    df["year"] = df[group_col].astype(str).str[:4].astype(int)
    df["kaisaikai"] = df[group_col].astype(str).str[4:6].astype(int)
    df["year_meet"] = df["year"].astype(str) + "-" + df["kaisaikai"].astype(str).str.zfill(2)

    # 基準年データ
    base_df = df[df["year"] == target_year]
    if base_df.empty:
        raise ValueError(f"{target_year}年のデータがありません。")

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in [group_col, "year", "kaisaikai", "year_meet"] and df[c].dtype != "object"
        ]

    dist_list = []

    # 年×開催ごとに比較
    for (yr, meet), df_group in tqdm(df.groupby(["year", "kaisaikai"]), disable=not verbose):
        if yr == target_year:
            continue

        total_dist = 0
        valid_feats = 0

        for feat in feature_cols:
            base_vals = base_df[feat].dropna()
            grp_vals = df_group[feat].dropna()
            if base_vals.empty or grp_vals.empty:
                continue

            # 数値特徴量
            if np.issubdtype(base_vals.dtype, np.number):
                bins = np.histogram_bin_edges(np.concatenate([base_vals, grp_vals]), bins=20)
                p_hist = np.histogram(base_vals, bins=bins, density=True)[0] + 1e-9
                q_hist = np.histogram(grp_vals, bins=bins, density=True)[0] + 1e-9

                if distance_type == "kl":
                    dist = entropy(p_hist, q_hist)
                elif distance_type == "js":
                    m = 0.5 * (p_hist + q_hist)
                    dist = 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))
                elif distance_type == "wasserstein":
                    # Wassersteinは値そのものを使う
                    dist = wasserstein_distance(base_vals, grp_vals)
                else:
                    raise ValueError(f"distance_type={distance_type} は未対応")
            else:
                # カテゴリ特徴量
                base_probs = base_vals.value_counts(normalize=True)
                grp_probs = grp_vals.value_counts(normalize=True)
                all_cats = set(base_probs.index) | set(grp_probs.index)
                p = np.array([base_probs.get(c, 1e-9) for c in all_cats])
                q = np.array([grp_probs.get(c, 1e-9) for c in all_cats])

                if distance_type == "kl":
                    dist = entropy(p, q)
                elif distance_type == "js":
                    m = 0.5 * (p + q)
                    dist = 0.5 * (entropy(p, m) + entropy(q, m))
                elif distance_type == "wasserstein":
                    # カテゴリに対してはKL/JS推奨
                    dist = np.nan
                else:
                    raise ValueError(f"distance_type={distance_type} は未対応")

            if not np.isnan(dist):
                total_dist += dist
                valid_feats += 1

        if valid_feats > 0:
            avg_dist = total_dist / valid_feats
            dist_list.append({
                "year": yr,
                "kaisaikai": meet,
                "year_meet": f"{yr}-{meet:02d}",
                "distance": avg_dist
            })

    dist_df = pd.DataFrame(dist_list).sort_values("distance", ascending=True)

    # 閾値以下の開催を抽出
    selected_meets = dist_df.loc[dist_df["distance"] <= threshold, "year_meet"].tolist()
    selected_df = df[df["year_meet"].isin(selected_meets)].drop(columns=["year", "kaisaikai", "year_meet"])

    if verbose:
        if len(selected_meets) > 0:
            print(f"\n✅ {target_year}年に分布が近い開催（{distance_type} ≤ {threshold}）:")
            for _, row in dist_df.loc[dist_df["year_meet"].isin(selected_meets)].iterrows():
                print(f"  - {row['year_meet']}: distance={row['distance']:.4f}")
        else:
            print(f"\n⚠️ 距離が {threshold} 以下の開催は見つかりませんでした。")

    return selected_df, dist_df

def time_series_group_split_by_year(df, year=2024, group_col="レースID"):
    """
    年ごとにtrain/val/testを分割する（時系列順保持）。
    
    - train: 2023年より前のデータ
    - val:   2023年データ
    - test:  2024年および2025年データ
    """

    # --- 年を抽出 ---
    years = df[group_col].astype(str).str[:4]

    # --- 各年のデータを分離 ---
    df_train = df[years.astype(int) < year].reset_index(drop=True)
    df_val   = df[years.astype(int) == year].reset_index(drop=True)
    df_test  = df[years.isin([str(year+1)])].reset_index(drop=True)

    # --- データ数チェック ---
    print("✅ データ分割完了")
    print(f"train年: <{year} ({df_train.shape[0]}件)")
    print(f"val年:   {year} ({df_val.shape[0]}件)")
    print(f"test年:  {year+1} ({df_test.shape[0]}件)")

    return df_train, df_val, df_test

'''
seed22
[top評価結果test ブートストラップ評価]
レース数: 775
的中率: 12.00%（95%CI: 9.81% ～ 14.32%）
回収率: 105.91%（95%CI: 77.20% ～ 138.58%）
'''

pairwise_list = [
    # 'hinge',
    'squared_hinge',
    # 'logistic',
    # 'bpr',
    # 'exp',
    # 'soft_margin',
    # 'tanh',
]

weight_mode_list = [
    # 'value_i',
    # 'roi',
    # 'abs_value_diff',
    'ev_i',
    # 'softmax_ev',
    # 'rank_focus',
    # 'focal_roi',
    # 'odds_aware',
]

if __name__ == '__main__':
    print(device)

    seed = 4
    set_seed(seed)  # 先に乱数固定
    fold_results = []

    # === 5. KFold処理 ===
    print(df.columns.values)
    
    # ラベル作成
    df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())
    df['着順'] = df['着順'].fillna(0)
    # df['人気'] = df['人気'].fillna(df['人気'].median())
    df['smooth_rel'] = make_smooth_relevance_labels(df)
    # df = make_rank_labels(df)

    # 出走頭数ビン
    # bins_horses = [0, 13, 16, 100]
    # labels_horses = ['small', 'medium', 'large']
    # df['num_horses_bin'] = pd.cut(df['出走頭数'], bins=bins_horses, labels=labels_horses)

    # 反転
    # df = inversion(df)

    # # カラム追加
    # df = append_col(df)
    # df = add_relative_features(df)
    

    # gkf = GroupKFold(n_splits=n_splits)
    # for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df[group_col])):
    # --- 使用例 ---
    # splits, df_test_2025, df = time_series_group_cv_3split_2025(df)
    random.seed(1)                    
    np.random.seed(1)
    splits, df = time_series_group_cv_3split_2025(df)

    # train_df, val_df, test_df = time_series_train_val_split(df, group_col="レースID", train_ratio=0.8)
    # print(val_df["レースID"].min())
    # print(val_df["レースID"].max())
    candidate_cols = ['齢', '間隔', '父馬', '騎手', '性', '齢']

    combo_list = None
    year = 2025

    # for seed in range(4, 400):
    for pairwise, weight_mode in itertools.product(pairwise_list, weight_mode_list):
        # if pairwise == 'hinge' and weight_mode != 'focal_roi': continue
        # elif pairwise == 'hinge' and weight_mode != 'odds_aware': continue
        print("seed:", seed)
        print("pairwise:", pairwise)
        print("weight_mode:", weight_mode)
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
            if fold != 2:
                continue
            
        # for fold in range(1): 
            train_df = df.loc[train_idx]
            val_df = df.loc[val_idx]
            test_df = df.loc[test_idx]
            year = year - 1
            # train_df, val_df, test_df = time_series_group_split_by_year(df, year=year)
            print("fold_num:", len(train_df))
            print("fold_num:", len(val_df))
            print("fold_num:", len(test_df))
            # print("fold_num:", len(df_test_2025))


            # trainval: test = 8 : 2（group単位）
            # trainval_df = df.iloc[train_idx]
            # test_df = df.iloc[test_idx]

            # # train:valid = 6 : 2（group単位）
            # gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)  # 0.75 of 8割 = 6割
            # train_idx, val_idx = next(gss.split(trainval_df, groups=trainval_df[group_col]))

            # train_df = trainval_df.iloc[train_idx]
            # val_df = trainval_df.iloc[val_idx]

            # print("fold_num:", len(train_df))

        # ---- 外側: trainval/test = 8:2 ----
        # sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

        # for fold, (trainval_idx, test_idx) in enumerate(
        #     sgkf.split(df, y=df["場所"], groups=df[group_col])
        # ):
        #     # if fold == 0:
        #     #     continue
        #     trainval_df = df.iloc[trainval_idx]
        #     test_df = df.iloc[test_idx]

        #     # ---- 内側: train/val = 6:2 (trainvalの中で) ----
        #     sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        #     inner_train_idx, val_idx = next(
        #         sgkf_inner.split(trainval_df, y=trainval_df["場所"], groups=trainval_df[group_col])
        #     )

        #     train_df = trainval_df.iloc[inner_train_idx]
        #     val_df = trainval_df.iloc[val_idx]

            # # 予測順位ごとの勝率
            # win_stats = train_df.groupby('pred_rank').apply(
            #     lambda x: (x['着順'] == 1).sum() / max(len(x), 1)
            # ).reset_index(name='win_prob')

            # # train_df にマージ
            # train_df = train_df.merge(win_stats, on='pred_rank', how='left')

            # # val_df にマージ
            # val_df = val_df.merge(win_stats, on='pred_rank', how='left')

            # # test_df にマージ
            # test_df = test_df.merge(win_stats, on='pred_rank', how='left')

            # # 条件付き統計（出走頭数bin × 予想順位）
            # group_cols = ['num_horses_bin', 'pred_rank']
            # win_stats = train_df.groupby(group_cols).apply(
            #     lambda x: (x['着順'] == 1).sum() / max(len(x), 1)
            # ).reset_index(name='win_prob')


            # # === dfに勝率をマージ ===
            # train_df = train_df.merge(
            #     win_stats,
            #     how='left',
            #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
            # )

            # # === dfに勝率をマージ ===
            # val_df = val_df.merge(
            #     win_stats,
            #     how='left',
            #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
            # )

            # # === dfに勝率をマージ ===
            # test_df = test_df.merge(
            #     win_stats,
            #     how='left',
            #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
            # )

            # # 5. 最終的な win_prob を追加
            # train_df['win_prob'] = train_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化
            # val_df['win_prob'] = val_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化
            # test_df['win_prob'] = test_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化

            # === 6. 特徴量エンコーディング ===
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)
            df_2025  = test_df.reset_index(drop=True)

            # train_df = build_train_features(train_df)
            # val_df, add_cols = apply_val_test_features(train_df, val_df)
            # test_df, _ = apply_val_test_features(train_df, test_df)
            # df_2025, _ = apply_val_test_features(train_df, df_2025)

            # print(val_df['複勝払戻'].head(50))

            # df_2025  = df_test_2025.reset_index(drop=True)

            # winrate_mapping, combo_list, feature_list = generate_stat_features_by_course_train(train_df, candidate_cols, n_sample=10, combo_list=combo_list)

            # train_df, feature_list = apply_stats_from_mapping(train_df, winrate_mapping)
            # val_df, feature_list = apply_stats_from_mapping(val_df, winrate_mapping)
            # test_df, feature_list = apply_stats_from_mapping(test_df, winrate_mapping)
            # df_2025, feature_list = apply_stats_from_mapping(df_2025, winrate_mapping)

            # scale_cols.extend(feature_list)

            
            train_df, sire_mapping = target_encoding(train_df, '父馬', target_col)
            # with open(f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl', "wb") as dd:
            #     pickle.dump(sire_mapping, dd)

            # val/test は train 全体の mapping を使う
            val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
            test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)
            df_2025['父馬_te'] = df_2025['父馬'].map(sire_mapping).fillna(-1)

            train_df, j_mapping = target_encoding(train_df, '騎手', target_col)
            # with open(f'./pickle-dict/jwin_dict_{field}_fold{fold}.pkl', "wb") as dd:
            #     pickle.dump(j_mapping, dd)

            # val/test は train 全体の mapping を使う
            val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
            test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)
            df_2025['騎手_te'] = df_2025['騎手'].map(j_mapping).fillna(-1)

            feature_cols = [col for col in df.columns if col not in ['オッズ',"複勝_hit_max", "複勝払戻_max", "複勝払戻", "複勝_hit", '人気', '馬番', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

            # zero_var = train_df[scale_cols].std()[train_df[scale_cols].std() == 0]
            # print("分散ゼロの列:", zero_var.index.tolist())

            # print("train shape:", train_df[scale_cols].shape)
            # print("val shape:", val_df[scale_cols].shape)
            # print("test shape:", test_df[scale_cols].shape)

            # print("NaN 含有数:\n", train_df[scale_cols].isna().sum())
            # print("有効サンプル数:", train_df[scale_cols].notna().sum())

            # === 7. 特徴量スケーリング ===
            scaler = StandardScaler()
            train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
            val_df[scale_cols] = scaler.transform(val_df[scale_cols])
            test_df[scale_cols] = scaler.transform(test_df[scale_cols])
            df_2025[scale_cols] = scaler.transform(df_2025[scale_cols])

            # train_df[scale_cols+add_cols] = scaler.fit_transform(train_df[scale_cols+add_cols])
            # val_df[scale_cols+add_cols] = scaler.transform(val_df[scale_cols+add_cols])
            # test_df[scale_cols+add_cols] = scaler.transform(test_df[scale_cols+add_cols])
            # df_2025[scale_cols+add_cols] = scaler.transform(df_2025[scale_cols+add_cols])

            # スケーラーを保存（モデルと同じディレクトリに置くのが一般的）
            joblib.dump(scaler, f"./model/scaler_{field}_fold{fold}.pkl")
            # joblib.dump(scale_cols, f"./pickle-dict/scal_cols.pkl")

            # === 0. データの前処理 ===
            # Nanの処理
            # joblib.dump(feature_cols, f"./pickle-dict/feature_cols_nan.pkl")
            train_df, val_df, test_df, df_2025 = fill_nan(train_df, feature_cols), fill_nan(val_df, feature_cols), fill_nan(test_df, feature_cols), fill_nan(df_2025, feature_cols)
            # カテゴリ変換
            
            train_df, map_dict = race_feature_train(train_df)
            val_df = race_feature_test(val_df, map_dict)
            test_df = race_feature_test(test_df, map_dict)
            df_2025 = race_feature_test(df_2025, map_dict)

            train_df = train_df.round(13)
            val_df = val_df.round(13)
            test_df = test_df.round(13)
            df_2025 = df_2025.round(13)

            # 保存
            # joblib.dump(map_dict, f"./pickle-dict/category_mappings_{field}_fold{fold}.pkl")

            # print(val_df.head(30))


            bad_vals = ~np.isfinite(train_df.select_dtypes(include=[np.number]))
            # print(bad_vals.sum())           # 各列ごとの個数

            # === 3. ランキング学習 ===
            # embedding_cols = feature_category + diff_category_place + diff_category_field
            # labmdarank_lgb(train_df, val_df, test_df, feature_cols, target_col, embedding_cols, fold)
            # continue

            # === 1. データの前提 ===
            # embedding_cols = feature_category + diff_category_place + diff_category_field
            embedding_cols = feature_category

            feature_cols = [col for col in feature_cols if col not in embedding_cols and col not in common_cols]
            # feature_cols = [col for col in feature_cols if col not in add_cols]

            # print(f"feature_cols: {feature_cols}")
            
            # feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
            # embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
            # context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
            # context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

            # joblib.dump(feature_cols, "./pickle-dict/feature_cols.pkl")
            # joblib.dump(embedding_cols, "./pickle-dict/embedding_cols.pkl")
            # joblib.dump(context_num_cols, "./pickle-dict/context_num_cols.pkl")
            # joblib.dump(context_cat_cols, "./pickle-dict/context_cat_cols.pkl")

            # print(f"feature_cols: {feature_cols}")
            # print(f"embedding_cols: {embedding_cols}")
            # print(f"context_num_cols: {context_num_cols}")
            # print(f"context_cat_cols: {context_cat_cols}")
            # continue
            

            X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups, win_index_train_groups = group_by_race(train_df)
            X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups, win_index_val_groups = group_by_race(val_df)
            X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups, win_index_test_groups = group_by_race(test_df)
            X_2025_groups, y_2025_groups, cat_2025_groups, context_2025_num_groups, context_2025_cat_groups, win_2025_groups, payout_2025_groups, win_index_2025_groups = group_by_race(df_2025)

            train_dataset = RaceDataset(X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups, win_index_train_groups)
            val_dataset = RaceDataset(X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups, win_index_val_groups)
            test_dataset = RaceDataset(X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups, win_index_test_groups)
            test_2025_dataset = RaceDataset(X_2025_groups, y_2025_groups, cat_2025_groups, context_2025_num_groups, context_2025_cat_groups, win_2025_groups, payout_2025_groups, win_index_2025_groups)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=set_seed(seed), generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_2025_loader = DataLoader(test_2025_dataset, batch_size=1, shuffle=False)

            all_df = pd.concat([train_df, val_df, test_df, df_2025], axis=0)
            embedding_sizes = []
            for col in embedding_cols:
                n_unique = all_df[col].nunique()
                n_unique = max(n_unique, 1)  # 定数列や全NaN列でも最低1
                embedding_sizes.append(n_unique + 5)  # 余裕分 +5

            context_embedding_sizes = []
            for col in context_cat_cols:
                n_unique = all_df[col].nunique()
                n_unique = max(n_unique, 1)
                context_embedding_sizes.append(n_unique + 5)

            # embedding_sizes = [train_df[col].nunique() + 5 for col in embedding_cols]  # 各カテゴリ列のクラス数
            # context_embedding_sizes = [train_df[col].nunique() + 5 for col in context_cat_cols]  # 各カテゴリ列のクラス数

            # モデル
            emb_dim = 64  
            model = ListNet(embedding_sizes=embedding_sizes, num_features=len(feature_cols), context_embedding_sizes=context_embedding_sizes, context_num_sizes=len(context_num_cols), emb_dim=emb_dim)
            model.apply(init_weights)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
            # ハイパーパラメータ
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(
            #     optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs
            # )
            
            patience = 10  # 何エポック改善がなければ終了するか
            best_val_loss = float('inf')
            best_roi = 0
            best_ndcg = 0
            no_improve_count = 0
            best_model_weights = None
            mse_loss_fn = nn.MSELoss()
            alpha = 0  # MSE の比率
            val_records = val_df.copy()
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in train_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    y_sum = y.detach().cpu().numpy().sum()
                    # ランク損失
                    preds = model(X, cat_X, context_X, context_cat_X)
                    # loss = lambdarank_loss(preds, y)
                    # loss = combined_loss(preds, y, gain, win_labels)
                    loss = combined_loss(preds, y, gain, win_labels, pairwise, weight_mode)
                    # loss = race_cross_entropy_loss(preds, win_labels)

                    # if winner >= 0:  # 正常なレースのみ
                    #     loss = F.cross_entropy(preds.unsqueeze(0), winner.unsqueeze(0))

                    # 勾配計算
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    total_loss += loss.item()
                # print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}")

                # Validation
                model.eval()
                val_loss = 0.0
                box = []
                with torch.no_grad():
                    for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in val_loader:
                        X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                        preds = model(X, cat_X, context_X, context_cat_X)
                        box.append(preds.squeeze().cpu().numpy())
                        # loss = listnet_loss(preds, y)
                        # loss = race_cross_entropy_loss(preds, win_labels)
                        # loss = combined_loss(preds, y, gain, win_labels)
                        loss = combined_loss(preds, y, gain, win_labels, pairwise, weight_mode)
                        # if winner >= 0:  # 正常なレースのみ
                        #     loss = F.cross_entropy(preds.unsqueeze(0), winner.unsqueeze(0))
                        # print(preds.mean().item(), preds.std().item())
                        # print(gain)

                        val_loss += loss.item()

                        
                
                val_records['pred_score'] = np.concatenate(box)
                # ndcg = calc_mean_ndcg(val_records)
                # print(f"Epoch {epoch+1}: NDCG: {ndcg:.4f}")

                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                # val_records['expected_value'] = val_records['pred_score'] * val_records['オッズ']
                # top = val_records.loc[val_records.groupby('レースID')['expected_value'].idxmax()]

                # # ブートストラップ
                # n_boot = 10000  # ブートストラップ試行回数
                # roi_list = []
                # acc_list = []

                # for _ in range(n_boot):
                #     # レース単位でリサンプリング（復元抽出）
                #     sampled = top.sample(frac=1.0, replace=True)
                    
                #     total_bet = len(sampled) * 100
                #     total_return = sampled["複勝払戻"].sum()  # 的中時のみ払戻あり
                    
                #     hit_count = sampled["複勝_hit"].sum()
                #     roi = total_return / total_bet
                #     acc = hit_count / len(sampled)
                    
                #     roi_list.append(roi)
                #     acc_list.append(acc)

                # roi_arr = np.array(roi_list)
                # acc_arr = np.array(acc_list)

                # # 点推定
                # mean_roi = roi_arr.mean()
                # mean_acc = acc_arr.mean()

                # # 95%信頼区間
                # roi_ci = np.percentile(roi_arr, [2.5, 97.5])
                # acc_ci = np.percentile(acc_arr, [2.5, 97.5])

                # print(f"\n[val ブートストラップ評価]")
                # print(f"レース数: {len(top)}")
                # print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
                # print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early Stopping 判定
                if avg_val_loss < best_val_loss - 0.1:
                    best_val_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # ベストモデルに戻す
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

            # model.eval()
            # all_preds = []
            # with torch.no_grad():
            #     for X, y, cat_X, context_X, context_cat_X in val_loader:
            #         X, y, cat_X, context_X, context_cat_X = (
            #             X[0].to(device),
            #             y[0].to(device),
            #             cat_X[0].to(device),
            #             context_X[0].to(device),
            #             context_cat_X[0].to(device)
            #         )
            #         preds = model(X, cat_X, context_X, context_cat_X)  # shape: (馬数,) または (馬数, 1)
            #         preds = preds.squeeze()  # 余分な次元がある場合に対応
            #         prob = preds.cpu().numpy()
            #         # prob = torch.softmax(preds, dim=0).cpu().numpy()
            #         all_preds.append(prob)

            # valでの出力スコアを集める
            val_preds = []
            model.eval()
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in val_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    preds = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    val_preds.append(preds)


            # ------------------------
            # Test評価
            # ------------------------
            test_scores = []
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in test_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    raw_pred = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    test_scores.append(raw_pred)

            test_2025_scores = []
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in test_2025_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    raw_pred = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    test_2025_scores.append(raw_pred)

            # スコア付与
            val_df = val_df.copy()
            test_df = test_df.copy()
            df_2025 = df_2025.copy()
            test_df['pred_score'] = np.concatenate(test_scores)
            val_df['pred_score'] = np.concatenate(val_preds)
            df_2025['pred_score'] = np.concatenate(test_2025_scores)

            test_df['expected_value'] = test_df['pred_score'] * test_df['オッズ']
            top = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]

            # ブートストラップ
            n_boot = 10000  # ブートストラップ試行回数
            roi_list = []
            acc_list = []

            for _ in range(n_boot):
                # レース単位でリサンプリング（復元抽出）
                sampled = top.sample(frac=1.0, replace=True)
                
                total_bet = len(sampled) * 100
                # total_return = sampled["複勝払戻"].sum()  # 的中時のみ払戻あり
                total_return = sampled["単勝オッズ"].sum()
                
                # hit_count = sampled["複勝_hit"].sum()
                hit_count = sampled["is_win"].sum()
                roi = total_return / total_bet
                acc = hit_count / len(sampled)
                
                roi_list.append(roi)
                acc_list.append(acc)

            roi_arr = np.array(roi_list)
            acc_arr = np.array(acc_list)

            # 点推定
            mean_roi = roi_arr.mean()
            mean_acc = acc_arr.mean()

            # 95%信頼区間
            roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            print(f"\n[ex評価結果test ブートストラップ評価]")
            print(f"レース数: {len(top)}")
            print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # ndcg = calc_mean_ndcg(test_df)
            # print(f"embedding_dim={emb_dim}, NDCG={ndcg:.4f}")

            # print(test_df[['pred_score', 'オッズ', 'expected_value']].sort_values('expected_value', ascending=False).head(20))
            # print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])

            # df_2025['expected_value'] = df_2025['pred_score'] * df_2025['オッズ']
            # top = df_2025.loc[df_2025.groupby('レースID')['expected_value'].idxmax()]

            # # ブートストラップ
            # n_boot = 10000  # ブートストラップ試行回数
            # roi_list = []
            # acc_list = []

            # for _ in range(n_boot):
            #     # レース単位でリサンプリング（復元抽出）
            #     sampled = top.sample(frac=1.0, replace=True)
                
            #     total_bet = len(sampled) * 100
            #     total_return = sampled["単勝オッズ"].sum()  # 的中時のみ払戻あり
                
            #     hit_count = sampled["is_win"].sum()
            #     roi = total_return / total_bet
            #     acc = hit_count / len(sampled)
                
            #     roi_list.append(roi)
            #     acc_list.append(acc)

            # roi_arr = np.array(roi_list)
            # acc_arr = np.array(acc_list)

            # # 点推定
            # mean_roi = roi_arr.mean()
            # mean_acc = acc_arr.mean()

            # # 95%信頼区間
            # roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            # acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            # print(f"\n[ex評価結果2025 ブートストラップ評価]")
            # print(f"レース数: {len(top)}")
            # print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            # print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # ndcg = calc_mean_ndcg(test_df)
            # print(f"embedding_dim={emb_dim}, NDCG={ndcg:.4f}")

            # top = df_2025.loc[df_2025.groupby('レースID')[f'pred_score'].idxmax()]

            # # ブートストラップ
            # n_boot = 10000  # ブートストラップ試行回数
            # roi_list = []
            # acc_list = []

            # for _ in range(n_boot):
            #     # レース単位でリサンプリング（復元抽出）
            #     sampled = top.sample(frac=1.0, replace=True)
                
            #     total_bet = len(sampled) * 100
            #     total_return = sampled["単勝オッズ"].sum()  # 的中時のみ払戻あり
                
            #     hit_count = sampled["is_win"].sum()
            #     roi = total_return / total_bet
            #     acc = hit_count / len(sampled)
                
            #     roi_list.append(roi)
            #     acc_list.append(acc)

            # roi_arr = np.array(roi_list)
            # acc_arr = np.array(acc_list)

            # # 点推定
            # mean_roi = roi_arr.mean()
            # mean_acc = acc_arr.mean()

            # # 95%信頼区間
            # roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            # acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            # print(f"\n[top評価結果2025 ブートストラップ評価]")
            # print(f"レース数: {len(top)}")
            # print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            # print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # print(f"\n[top評価結果2025]")
            # print(f"レース数: {len(top)}")
            # print(f"的中数: {int(hit_count)}")
            # print(f"的中率: {hit_count / len(top):.2%}")
            # print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

            top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]
            # top = top[top['pred_score'] * top['オッズ'] > 1.0]

            # ブートストラップ
            n_boot = 10000  # ブートストラップ試行回数
            roi_list = []
            acc_list = []

            for _ in range(n_boot):
                # レース単位でリサンプリング（復元抽出）
                sampled = top.sample(frac=1.0, replace=True)
                
                total_bet = len(sampled) * 100
                total_return = sampled["複勝払戻"].sum()  # 的中時のみ払戻あり
                total_return = sampled["単勝オッズ"].sum()  # 的中時のみ払戻あり
                
                # hit_count = sampled["複勝_hit"].sum()
                hit_count = sampled["is_win"].sum()
                roi = total_return / total_bet
                acc = hit_count / len(sampled)
                
                roi_list.append(roi)
                acc_list.append(acc)

            roi_arr = np.array(roi_list)
            acc_arr = np.array(acc_list)

            # 点推定
            mean_roi = roi_arr.mean()
            mean_acc = acc_arr.mean()

            # 95%信頼区間
            roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            print(f"\n[top評価結果test ブートストラップ評価]")
            print(f"レース数: {len(top)}")
            print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # 上のコードそのまま
            # selected = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]
            # top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]

            # # per-race で return 計算
            # selected['bet_return'] = np.where(selected['着順'] == 1, selected['単勝オッズ'], 0.0) - 1.0
            # top['bet_return'] = np.where(top['着順'] == 1, top['単勝オッズ'], 0.0) - 1.0

            # # foldごとに保存
            # fold_results.append({
            #     'fold': fold,
            #     'selected_returns': selected[['レースID', 'bet_return']],
            #     'top_returns': top[['レースID', 'bet_return']],
            # })

            # val_df.to_csv(f'./csv/{field}_result_ranknet3_val_{fold}.csv', index=False)
            # test_df.to_csv(f'./csv/{field}_result_ranknet_test_fuku_{fold}.csv', index=False)
            test_df.to_csv(f'./csv/{field}_result_ranknet_test_{fold}.csv', index=False)
            # df_2025.to_csv(f'./csv/{field}_result_ranknet3_2025_{fold}.csv', index=False)
            # if mean_roi > 1.0:
            #     with open("log.txt", "a", encoding="utf-8") as f:
            #         f.write(f"seed:{seed}")
            #         f.write(f"roi:{mean_roi}")
            # with open("log.txt", "a", encoding="utf-8") as f:
            #     f.write(f"field:{field}\n")
            #     f.write(f"seed:{seed}\n")
            #     f.write(f"pairwise:{pairwise}\n")
            #     f.write(f"weight_mode:{weight_mode}\n")
            #     f.write(f"roi:{mean_roi}\n\n")
                # sys.exit()
                # seed47


            # モデルを保存
            torch.save(model.state_dict(), f'./model/{field}_ranknet_{fold}.pth')

        # df_all = pd.concat([
        #     r['selected_returns'].assign(fold=r['fold'], policy='expected')
        #     for r in fold_results
        #     ] + [
        #         r['top_returns'].assign(fold=r['fold'], policy='top')
        #         for r in fold_results
        #     ], ignore_index=True)

        # # 集計
        # summary = df_all.groupby(['fold', 'policy'])['bet_return'].mean().unstack()
        # print(summary)
        # print("平均ROI (per fold):")
        # print(summary.mean())
        # print("標準偏差:")
        # print(summary.std())

        # # ペア検定 (fold単位)
        # t_stat, p_value = stats.ttest_rel(summary['expected'], summary['top'])
        # print(f"T-test: stat={t_stat:.3f}, p={p_value:.4f}")

        # # Wilcoxon (より頑健)
        # stat, p = stats.wilcoxon(summary['expected'], summary['top'])
        # print(f"Wilcoxon: stat={stat:.3f}, p={p:.4f}")

        # ['着順' '馬番' '斤量' '騎手' '人気' '単勝オッズ' '距離' 'フィールド' '馬場' '出走頭数' '馬単' 'レースID'
        #  '父馬' '間隔' '性' '齢' '1場所' '1過去着順' '1フィールド' '1距離' '1タイム' '1馬場' '1出走馬数' '1馬番'
        #  '1人気' '1斤量' '1コーナー通過順' '1後3F' '1馬体重' '1体重増減' '1着差' '1クラス' '1スピード指数'
        #  '1距離差' '1場所変化' '1フィールド変化' '2場所' '2過去着順' '2フィールド' '2距離' '2タイム' '2馬場'
        #  '2出走馬数' '2馬番' '2人気' '2斤量' '2コーナー通過順' '2後3F' '2馬体重' '2体重増減' '2着差' '2クラス'
        #  '2スピード指数' '2距離差' '2場所変化' '2フィールド変化' '3場所' '3過去着順' '3フィールド' '3距離' '3タイム'
        #  '3馬場' '3出走馬数' '3馬番' '3人気' '3斤量' '3コーナー通過順' '3後3F' '3馬体重' '3体重増減' '3着差'
        #  '3クラス' '3スピード指数' '3距離差' '3場所変化' '3フィールド変化' '4場所' '4過去着順' '4フィールド' '4距離'
        #  '4タイム' '4馬場' '4出走馬数' '4馬番' '4人気' '4斤量' '4コーナー通過順' '4後3F' '4馬体重' '4体重増減'
        #  '4着差' '4クラス' '4スピード指数' '4距離差' '4場所変化' '4フィールド変化' '5場所' '5過去着順' '5フィールド'
        #  '5距離' '5タイム' '5馬場' '5出走馬数' '5馬番' '5人気' '5斤量' '5コーナー通過順' '5後3F' '5馬体重'
        #  '5体重増減' '5着差' '5クラス' '5スピード指数' '5距離差' '5場所変化' '5フィールド変化' '平均クラス' '平均ペース'
        #  '1クラス差' '1ペース差' 'best着差' 'bestスピード指数' 'av着差' 'avスピード指数' '上昇度' 'オッズ'
        #  'rank']

        # feature_category = '距離', 'フィールド', '馬場', '出走頭数', '馬番', '性',
        # '1場所', '2場所', '3場所', '4場所', '5場所', '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド',
        # '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド', '1距離', '2距離', '3距離', '4距離', '5距離',
        # '1馬場', '2馬場', '3馬場', '4馬場', '5馬場','1コーナー通過順', '2コーナー通過順', '3コーナー通過順', '4コーナー通過順', '5コーナー通過順',
        # '1斤量', '2斤量', '3斤量', '4斤量', '5斤量', '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数', '1馬番', '2馬番', '3馬番', '4馬番', '5馬番',

