import random
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as lgb
import torch
import seaborn as sns

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)
# 小数点をすべて表示（指数表記なし）
pd.set_option('display.float_format', lambda x: f'{x:.16f}'.rstrip('0').rstrip('.'))

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

def add_pred_features(df, prefix="pred"):
    """
    複数の pred_score 列から統計特徴量を作成して DataFrame に追加する。

    Parameters
    ----------
    df : pd.DataFrame
        'pred_score_1', 'pred_score_2', ... のような列を含む DataFrame
    prefix : str
        新しく作る特徴量列のプレフィックス（例: 'pred' → 'pred_mean' など）

    Returns
    -------
    df : pd.DataFrame
        元の df に統計特徴量を追加した DataFrame
    """
    # pred_score系の列を抽出
    pred_cols = [c for c in df.columns if c.startswith("pred_score_")]
    if not pred_cols:
        raise ValueError("pred_score_ で始まる列が見つかりません。")

    # 各種特徴量を追加
    df[f"{prefix}_mean"]  = df[pred_cols].mean(axis=1)
    df[f"{prefix}_std"]   = df[pred_cols].std(axis=1)
    df[f"{prefix}_max"]   = df[pred_cols].max(axis=1)
    df[f"{prefix}_min"]   = df[pred_cols].min(axis=1)
    df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
    df[f"{prefix}_var"]   = df[pred_cols].var(axis=1)
    df[f"{prefix}_median"] = df[pred_cols].median(axis=1)
    df[f"{prefix}_skew"]   = df[pred_cols].skew(axis=1)
    df[f"{prefix}_kurt"]   = df[pred_cols].kurt(axis=1)

    return df

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

# def add_score_diff_features(df, no, target_col='pred_score'):
#     """
#     pred_score（モデルのスコア）を用いてレース内の相対的特徴量を追加する。
#     """
#     df = df.copy()
#     new_rows = []

#     for race_id, race in df.groupby('レースID'):
#         race = race.sort_values(target_col, ascending=False).reset_index(drop=True)
        
#         mean_score = race[target_col].mean()
#         std_score = race[target_col].std() if race[target_col].std() != 0 else 1e-6
#         min_score = race[target_col].min()
#         max_score = race[target_col].max()
#         score_range = max_score - min_score if max_score != min_score else 1e-6
        
#         # --- 差分・順位系 ---
#         race[f'{no}rank_in_race'] = range(1, len(race)+1)
#         race[f'{no}score_diff_prev'] = race[target_col].diff(-1)  # 下との差
#         race[f'{no}score_diff_next'] = race[target_col].diff()    # 上との差
#         race[f'{no}score_diff_top1'] = race[target_col].iloc[0] - race[target_col]
#         race[f'{no}score_diff_top3_mean'] = race[target_col].iloc[:3].mean() - race[target_col]
        
#         # --- 統計・分布系 ---
#         race[f'{no}score_mean'] = mean_score
#         race[f'{no}score_std'] = std_score
#         race[f'{no}score_range'] = score_range
#         race[f'{no}score_cv'] = std_score / (mean_score + 1e-6)
#         race[f'{no}score_minus_mean'] = race[target_col] - mean_score
#         race[f'{no}score_minus_mean_std'] = (race[target_col] - mean_score) / std_score
        
#         # --- 正規化・確率化 ---
#         race[f'{no}score_relative'] = (race[target_col] - min_score) / score_range
#         exp_score = np.exp(race[target_col] - race[target_col].max())  # 安定化
#         race[f'{no}score_softmax'] = exp_score / exp_score.sum()
#         race[f'{no}score_z'] = (race[target_col] - mean_score) / std_score

#         # --- 分布特性（レース単位） ---
#         score_softmax = race[f'{no}score_softmax'].values
#         entropy = -np.sum(score_softmax * np.log(score_softmax + 1e-6))
#         race[f'{no}score_entropy'] = entropy
#         race[f'{no}score_top_mean'] = race[target_col].iloc[:3].mean()
#         race[f'{no}score_bottom_mean'] = race[target_col].iloc[-3:].mean()
#         race[f'{no}score_top_bottom_diff'] = race[f'{no}score_top_mean'] - race[f'{no}score_bottom_mean']
#         race[f'{no}score_top_ratio'] = race[target_col] / (race[target_col].iloc[0] + 1e-6)
#         race[f'{no}score_rank_gap_ratio'] = race[f'{no}score_diff_prev'] / (race[target_col].abs() + 1e-6)

#         new_rows.append(race)

#     return pd.concat(new_rows, axis=0).reset_index(drop=True)

def load_csv(path):
    # 学習データを読み込む
    df = pd.read_csv(path, index_col=0)
    return df

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)

field = "monbetu"

csv_files = [
    f'./csv/{field}_result_stacking2_2025_{i}.csv' for i in range(5)
]

dfs = [pd.read_csv(f) for f in csv_files]
df = dfs[0][['レースID', '馬番', '単勝オッズ', 'is_win', 'オッズ']].copy()

# 各foldのスコアから順位を算出
for i, d in enumerate(dfs):
    df[f'pred_score_{i}'] = d['pred_score']

# val_df = df[df['レースID'].astype(str).str[:4] == "2024"].reset_index(drop=True)
# test_df = df[df['レースID'].astype(str).str[:4] != "2024"].reset_index(drop=True)

# group_col = "レースID" など、グループを示すカラム名を指定
group_col = "レースID"

# レースIDを昇順に並べる（古い→新しい）
race_ids = sorted(df[group_col].unique())

# 分割位置を決定（例：3:1）
split_idx = int(len(race_ids) * 0.75)

# val/test用のレースIDを取得
val_races = race_ids[:split_idx]
test_races = race_ids[split_idx:]

# データを分割（レース単位）
val_df = df[df[group_col].isin(val_races)].reset_index(drop=True)
test_df = df[df[group_col].isin(test_races)].reset_index(drop=True)


# train_df = add_pred_features(train_df)
val_df = add_pred_features(val_df).round(10)
test_df = add_pred_features(test_df).round(10)

# グルーピング
# train_df = train_df.sort_values(["レースID"]).reset_index(drop=True)
# train_list = train_df.groupby("レースID").size().to_list()
val_df = val_df.sort_values(["レースID"]).reset_index(drop=True)
eval_list = val_df.groupby("レースID").size().to_list()
test_df = test_df.sort_values(["レースID"]).reset_index(drop=True)
test_list = test_df.groupby("レースID").size().to_list()

# パラメータ設定
target_col = 'is_win'
feature_cols = [col for col in val_df.columns if col not in ['label', '馬番', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
print(feature_cols)
joblib.dump(feature_cols, "./pickle-dict/stacking_fold_feature_cols.pkl")
rate = 0.1
seed = 4 # 4
set_seed(seed)
lgb_train = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], group=eval_list)
lgb_eval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=lgb_train, group=test_list)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    # 'objective': 'regression',  # ←ここでランキング学習と指定！
    # 'metric': 'rmse',   # for lambdarank
    'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
    'learning_rate': rate,
    'random_state': seed,
    'verbose_eval': 1000,
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
    'label_gain': [0,3,5,10],
    'bagging_seed': seed,
    'feature_fraction_seed': seed,
    'data_random_seed': seed,
    'deterministic': True,        # LightGBM 3.3.0 以降で利用可能
    'force_col_wise': True,       # 再現性を高める（内部順序を固定）
    'num_threads': 1,             # 厳密再現のためスレッド固定
    'num_leaves': 30,             # ← ここで指定
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

best_params = tuner.best_params
model = tuner.get_best_booster()
print(best_params)

# pklファイルとしてモデルを保存
joblib.dump(model, f"./model/stacking_fold_model_lgb.pickle")

# 学習後のモデル
importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

feature_names = val_df[feature_cols].columns

feat_imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance_gain": importance_gain,
    "importance_split": importance_split
}).sort_values(by="importance_gain", ascending=False)

print(feat_imp_df)  # 上位20特徴量

# plt.figure(figsize=(10,6))
# sns.barplot(x="importance_gain", y="feature", data=feat_imp_df)
# plt.title("Top 20 Feature Importance (LambdaRank)")
# plt.show()

####### 第二学習 ########
# y_pred = model.predict(val_df[feature_cols])
# val_df['pred_score'] = y_pred

# # テストデータの予測 (予測クラスを返す)
# y_pred = model.predict(test_df[feature_cols])
# test_df['pred_score'] = y_pred

# val_df = add_score_diff_features(val_df)
# test_df = add_score_diff_features(test_df)

# feature_cols = [col for col in val_df.columns if col not in ['label', '馬番', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

# lgb_train = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], group=eval_list)
# lgb_eval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=lgb_train, group=test_list)

# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     # 'objective': 'regression',  # ←ここでランキング学習と指定！
#     # 'metric': 'rmse',   # for lambdarank
#     'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
#     'learning_rate': rate,
#     'random_state': seed,
#     'verbose_eval': 1000,
#     'objective': 'lambdarank',
#     'metric': 'ndcg',
#     'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
#     'label_gain': [0,3,5,10],
#     'bagging_seed': seed,
#     'feature_fraction_seed': seed,
#     'data_random_seed': seed,
#     'deterministic': True,        # LightGBM 3.3.0 以降で利用可能
#     'force_col_wise': True,       # 再現性を高める（内部順序を固定）
#     'num_threads': 1,             # 厳密再現のためスレッド固定
# }
# ####################################################################################


# tuner = lgb.LightGBMTuner(
#     params,
#     optuna_seed=seed,
#     train_set=lgb_train,
#     valid_sets=[lgb_eval],
#     # categorical_feature=cat_list,
#     early_stopping_rounds=20,  # ← ここで指定
#     num_boost_round=10000,      # ← イテレーション上限
#     callbacks=[lgb.log_evaluation(period=0)]
# )

# tuner.run()

# best_params = tuner.best_params
# model = tuner.get_best_booster()

# # pklファイルとしてモデルを保存
# # with open(f"{tuner_path}{seed}.pickle", "wb") as mk:
# #     pickle.dump(model, mk)

# # 学習後のモデル
# importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
# importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

# feature_names = val_df[feature_cols].columns

# feat_imp_df = pd.DataFrame({
#     "feature": feature_names,
#     "importance_gain": importance_gain,
#     "importance_split": importance_split
# }).sort_values(by="importance_gain", ascending=False)

# print(feat_imp_df)  # 上位20特徴量

# plt.figure(figsize=(10,6))
# sns.barplot(x="importance_gain", y="feature", data=feat_imp_df)
# plt.title("Top 20 Feature Importance (LambdaRank)")
# plt.show()


################# 評価 ##############

y_pred = model.predict(val_df[feature_cols])
val_df['pred_score'] = y_pred

# テストデータの予測 (予測クラスを返す)
y_pred = model.predict(test_df[feature_cols])
test_df['pred_score'] = y_pred

val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]
print("select len",len(selected))
total_bet = len(selected) * 100
total_return = selected['単勝オッズ'].sum()

hit_count = (selected['is_win'] == 1).sum()
roi = total_return / total_bet
 
print(f"\n[評価結果]")
print(f"レース数: {len(selected)}")
print(f"的中数: {int(hit_count)}")
print(f"的中率: {hit_count / len(selected):.2%}")
print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

test_df['expected_value'] = test_df['pred_score'] * test_df['オッズ']
selected = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]

total_bet = len(selected) * 100
total_return = selected['単勝オッズ'].sum()

hit_count = (selected['is_win'] == 1).sum()
roi = total_return / total_bet

print(f"\n[評価結果2025]")
print(f"レース数: {len(selected)}")
print(f"的中数: {int(hit_count)}")
print(f"的中率: {hit_count / len(selected):.2%}")
print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]


# # 1. 各レースで予想順位を付ける（スコアが高いほど1位）
# df_2025['pred_rank'] = df_2025.groupby('レースID')['pred_score'] \
#                             .rank(ascending=False, method='first')

# # 2. 各レースで上位3頭を抽出
# top3 = df_2025[df_2025['pred_rank'] <= 3].copy()

# # 3. 人気との乖離を計算
# # 人気は1が最も人気、数値が大きいほど低人気
# # → 値が大きいほど「予想より人気が低い」＝過小評価されている
# top3['pop_diff'] = top3['人気'] - top3['pred_rank']

# # 4. 各レースでpop_diffが最大の馬（市場が最も過小評価している馬）を抽出
# top = top3.loc[top3.groupby('レースID')['pop_diff'].idxmax()].reset_index(drop=True)

total_bet = len(top) * 100
total_return = top['単勝オッズ'].sum()

hit_count = (top['is_win'] == 1).sum()
roi = total_return / total_bet

print(f"\n[top評価結果2025]")
print(f"レース数: {len(top)}")
print(f"的中数: {int(hit_count)}")
print(f"的中率: {hit_count / len(top):.2%}")
print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]

# 1. 各レースで予想順位を付ける（スコアが高いほど1位）
# test_df['pred_rank'] = test_df.groupby('レースID')['pred_score'] \
#                             .rank(ascending=False, method='first')

# # 2. 各レースで上位3頭を抽出
# top3 = test_df[test_df['pred_rank'] <= 3].copy()

# # 3. 人気との乖離を計算
# # 人気は1が最も人気、数値が大きいほど低人気
# # → 値が大きいほど「予想より人気が低い」＝過小評価されている
# top3['pop_diff'] = top3['人気'] - top3['pred_rank']

# # 4. 各レースでpop_diffが最大の馬（市場が最も過小評価している馬）を抽出
# top = top3.loc[top3.groupby('レースID')['pop_diff'].idxmax()].reset_index(drop=True)

total_bet = len(top) * 100
total_return = top['単勝オッズ'].sum()

hit_count = (top['is_win'] == 1).sum()
roi = total_return / total_bet

print(f"\n[top評価結果]")
print(f"レース数: {len(top)}")
print(f"的中数: {int(hit_count)}")
print(f"的中率: {hit_count / len(top):.2%}")
print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

val_df.to_csv(f'./csv/{field}_result_stacking_fold_test.csv', index=False)
test_df.to_csv(f'./csv/{field}_result_stacking_fold_2025.csv', index=False)

