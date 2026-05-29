import numpy as np
import pandas as pd
import copy
import pickle
import random
import warnings
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import make_pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import Learning
import torch.nn.functional as F
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
from sklearn.model_selection import StratifiedGroupKFold
from src import Listwise as lw
import seaborn as sns

def add_score_diff_features(df):
    """
    pred_score（モデルのスコア）を用いてレース内の相対的特徴量を追加する。
    """
    df = df.copy()
    new_rows = []

    for race_id, race in df.groupby('レースID'):
        race = race.sort_values(
            by=['pred_score', '馬番'],  # 第二ソートキーを指定
            ascending=[False, True]     # pred_scoreは降順、馬番は昇順
        ).reset_index(drop=True)
        
        mean_score = race['pred_score'].mean()
        std_score = race['pred_score'].std() if race['pred_score'].std() != 0 else 1e-6
        min_score = race['pred_score'].min()
        max_score = race['pred_score'].max()
        score_range = max_score - min_score if max_score != min_score else 1e-6
        
        # --- 差分・順位系 ---
        race['rank_in_race'] = range(1, len(race)+1)
        race['score_diff_prev'] = race['pred_score'].diff(-1)  # 下との差
        race['score_diff_next'] = race['pred_score'].diff()    # 上との差
        race['score_diff_top1'] = race['pred_score'].iloc[0] - race['pred_score']
        race['score_diff_top3_mean'] = race['pred_score'].iloc[:3].mean() - race['pred_score']
        
        # --- 統計・分布系 ---
        race['score_mean'] = mean_score
        race['score_std'] = std_score
        race['score_range'] = score_range
        race['score_cv'] = std_score / (mean_score + 1e-6)
        race['score_minus_mean'] = race['pred_score'] - mean_score
        race['score_minus_mean_std'] = (race['pred_score'] - mean_score) / std_score
        
        # --- 正規化・確率化 ---
        race['score_relative'] = (race['pred_score'] - min_score) / score_range
        exp_score = np.exp(race['pred_score'] - race['pred_score'].max())  # 安定化
        race['score_softmax'] = exp_score / exp_score.sum()
        race['score_z'] = (race['pred_score'] - mean_score) / std_score

        # --- 分布特性（レース単位） ---
        score_softmax = race['score_softmax'].values
        entropy = -np.sum(score_softmax * np.log(score_softmax + 1e-6))
        race['score_entropy'] = entropy
        race['score_top_mean'] = race['pred_score'].iloc[:3].mean()
        race['score_bottom_mean'] = race['pred_score'].iloc[-3:].mean()
        race['score_top_bottom_diff'] = race['score_top_mean'] - race['score_bottom_mean']
        race['score_top_ratio'] = race['pred_score'] / (race['pred_score'].iloc[0] + 1e-6)
        race['score_rank_gap_ratio'] = race['score_diff_prev'] / (race['pred_score'].abs() + 1e-6)

        new_rows.append(race)

    return pd.concat(new_rows, axis=0).reset_index(drop=True)

def split_test_for_next_stage(test_df, group_col="レースID", ratios=(0.8, 0.2, 0.0)):
    """
    test_dfを次段モデル用にtrain/val/testに分割（時系列順）
    ratios=(train_ratio, val_ratio, test_ratio)
    """
    # レース単位で時系列順にソート
    unique_races = np.sort(test_df[group_col].unique())
    n = len(unique_races)
    
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    
    train_races = unique_races[:train_end]
    val_races = unique_races[train_end:val_end]
    test_races = unique_races[val_end:]
    
    train_idx = test_df[test_df[group_col].isin(train_races)].index
    val_idx = test_df[test_df[group_col].isin(val_races)].index
    test_idx = test_df[test_df[group_col].isin(test_races)].index
    
    return train_idx, val_idx, test_idx

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

field = "tokyo"
fold = 0
target_col = "ex"
seed = 1
set_seed(seed)

df = pd.read_csv(f'./csv/{field}_result_ranknet2_test_{fold}.csv')
df = df[['レースID', '馬番', '単勝オッズ', 'is_win', 'オッズ', 'pred_score']].copy()

df_2025 = pd.read_csv(f'./csv/{field}_result_ranknet2_2025_{fold}.csv')
df_2025 = df_2025[['レースID', '馬番', '単勝オッズ', 'is_win', 'オッズ', 'pred_score']].copy()

df['ex'] = df['pred_score'] * df['オッズ']
df_2025['ex'] = df_2025['pred_score'] * df_2025['オッズ']

df = add_score_diff_features(df)
df_2025 = add_score_diff_features(df_2025)

feature_cols = [col for col in df.columns if col not in ['レースID', '馬番', '単勝オッズ', 'is_win', 'オッズ', 'ex']]

train_idx, val_idx, test_idx = split_test_for_next_stage(df)

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]


# グルーピング
train_df = train_df.sort_values(["レースID"]).reset_index(drop=True)
train_list = train_df.groupby("レースID").size().to_list()
val_df = val_df.sort_values(["レースID"]).reset_index(drop=True)
eval_list = val_df.groupby("レースID").size().to_list()

# パラメータ設定
# cat_list = lw.feature_category + lw.diff_category_place + lw.diff_category_field + lw.context_cat_cols
rate = 0.01
lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], group=train_list)
lgb_eval = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], reference=lgb_train, group=eval_list)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',  # ←ここでランキング学習と指定！
    'metric': 'rmse',   # for lambdarank
    'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
    'learning_rate': rate,
    'random_state': seed,
    'verbose_eval': 1000,
    # 'objective': 'lambdarank',
    # 'metric': 'ndcg',
    # 'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
    # 'label_gain': [0,3,5,10],
    'bagging_seed': seed,
    'feature_fraction_seed': seed,
    'data_random_seed': seed,
    'deterministic': True,        # LightGBM 3.3.0 以降で利用可能
    'force_col_wise': True,       # 再現性を高める（内部順序を固定）
    'num_threads': 1,             # 厳密再現のためスレッド固定
}
####################################################################################


tuner = lgb.LightGBMTuner(
    params,
    optuna_seed=seed,
    train_set=lgb_train,
    valid_sets=[lgb_eval],
    # categorical_feature=cat_list,
    early_stopping_rounds=20,  # ← ここで指定
    num_boost_round=10000,      # ← イテレーション上限
    callbacks=[lgb.log_evaluation(period=0)]
)

tuner.run()
# get_best_booster() は使えないので best_params を取得する
best_params = tuner.best_params
model = tuner.get_best_booster()


# テストデータの予測 (予測クラスを返す)
y_pred = model.predict(df_2025[feature_cols])
df_2025[f'result{seed}'] = y_pred

top = df_2025.loc[df_2025.groupby('レースID')[f'result{seed}'].idxmax()]

# ブートストラップ
n_boot = 10000  # ブートストラップ試行回数
roi_list = []
acc_list = []

for _ in range(n_boot):
    # レース単位でリサンプリング（復元抽出）
    sampled = top.sample(frac=1.0, replace=True)
    
    total_bet = len(sampled) * 100
    total_return = sampled['単勝オッズ'].sum()  # 的中時のみ払戻あり
    
    hit_count = sampled['is_win'].sum()
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

print(f"\n[top評価結果2025 ブートストラップ評価]")
print(f"レース数: {len(top)}")
print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")
