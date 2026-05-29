import copy
import itertools
import pickle
import random
import warnings
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from src import listnet_config as cfg

# CSV読み込み
def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df

# レースごとにスコア最大の馬をランダムに1頭選択
def pick_top_random(df, score_col="pred_score", race_col="レースID"):
    def _pick(group):
        max_score = group[score_col].max()
        tied = group[group[score_col] == max_score]
        return tied.sample(n=1)
    return (
        df
        .groupby(race_col, group_keys=False)
        .apply(_pick)
        .reset_index(drop=True)
    )

# 条件別ROI統計量を計算
def compute_roi_stats(df, course_cols, target_cols, win_col='is_win', odds_col='単勝オッズ'):
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
        g["roi"] = g["payout"] / (g["n"] * 100)
        stats[col] = g[course_cols + [col, "roi"]]
    return stats

# ROI統計を特徴量としてマージ
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

# 学習データでROI特徴量を作成（2-fold交差内挿）
def build_train_features(df_train):
    course_cols = ['場所','距離','フィールド','馬場']
    target_cols = ['騎手','馬番','1距離','1場所','1フィールド']
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    df_train = df_train.copy()
    df_train['roi_feat_dummy'] = np.nan
    parts = []
    for (idx_a, idx_b) in kf.split(df_train):
        A = df_train.iloc[idx_a]
        B = df_train.iloc[idx_b]
        stats_A = compute_roi_stats(A, course_cols, target_cols)
        B2, _ = merge_roi_features(B, stats_A, course_cols, target_cols)
        stats_B = compute_roi_stats(B, course_cols, target_cols)
        A2, _ = merge_roi_features(A, stats_B, course_cols, target_cols)
        parts.append(A2)
        parts.append(B2)
    df_out = pd.concat(parts).sort_index()
    return df_out

# 検証/テストデータで学習データのROI統計量を適用
def apply_val_test_features(df_train, df_eval):
    course_cols = ['場所','距離','フィールド','馬場']
    target_cols = ['騎手','馬番','1距離','1場所','1フィールド']
    stats_train = compute_roi_stats(df_train, course_cols, target_cols)
    df_eval_out, add_cols = merge_roi_features(df_eval, stats_train, course_cols, target_cols)
    return df_eval_out, add_cols

# 同距離・同場所の過去成績を集計した履歴特徴を追加
def add_history_features(df):
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

    cfg.feature_cols.extend(['同距離過去率', '同場所過去率'])
    cfg.scale_cols.extend(['同距離過去率', '同場所過去率'])
    cfg.feature_category.extend(['同距離過去数', '同距離過去3着内数', '同場所過去数', '同場所過去3着内数'])

    return df

# 欠損値を-9999で埋める
def fill_nan(df, cols):
    df[cols] = df[cols].fillna(-9999)
    return df

# 過去のLightGBMランクモデルでスコア予測（評価用）
def eval_rank(df):
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', '1休養', '2休養', '3休養', '4休養', '5休養']]
    df['score'] = np.nan
    for fold, (train_idx, val_idx) in enumerate(kf.split(df['レースID'].unique())):
        train_races = df['レースID'].unique()[train_idx]
        val_races = df['レースID'].unique()[val_idx]
        train_data = df[df['レースID'].isin(train_races)]
        val_data = df[df['レースID'].isin(val_races)]
        with open(f'./pickle-tuner/tokyo_rank_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)
        df.loc[val_data.index, 'score'] = model.predict(val_data[feature_cols], num_iteration=model.best_iteration)

# カテゴリカラムをcategory型→コード変換（学習用・コード維持）
def race_feature(df):
    category = cfg.feature_category + cfg.context_cat_cols
    for col in category:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df

# カテゴリカラム変換（学習用・マッピング保存）
def race_feature_train(df):
    category_mappings = {}
    category = cfg.feature_category + cfg.context_cat_cols
    for col in category:
        df[col] = df[col].astype('category')
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes
    return df, category_mappings

# カテゴリカラム変換（テスト用・学習時のマッピングを参照）
def race_feature_test(df, category_mappings):
    category = cfg.feature_category + cfg.context_cat_cols
    for col in category:
        if col in df.columns:
            df[col] = df[col].astype(str)
            inv_map = {v:k for k,v in category_mappings[col].items()}
            key_type = type(list(inv_map.keys())[0])
            n_train_categories = len(category_mappings[col])
            df[col] = df[col].map(lambda x: inv_map.get(key_type(x), n_train_categories)).astype(int)
    return df

# 反転カラムを符号反転
def inversion(df):
    for col in cfg.inversion_cols:
        df[col] = -df[col]
    return df

# 通常のsoftmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# 順位（小さいほど良い）を反転したsoftmax確率
def softmax_neg_rank(rankings):
    x = -rankings
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# 滑らかな関連度ラベル: 着順に応じて[0, max_rel]
def make_smooth_relevance_labels(df, max_rel=3.0):
    return (max_rel / np.log1p(df['着順'] + 1)).clip(0, max_rel)

# 着順をレース内相対ランクに変換しbin分割で離散化
def make_rank_labels(df, group_col='レースID', pos_col='着順', n_bins=18):
    df = df.copy()
    rel_series = pd.Series(index=df.index, dtype=float)
    for race_id, g in df.groupby(group_col):
        n = len(g)
        rel_raw = (n - g[pos_col].values + 1) / n
        rel_series.loc[g.index] = rel_raw
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = np.digitize(rel_series.values, bins, right=True) - 1
    labels = np.clip(labels, 0, n_bins).astype(int)
    df['rank_label'] = labels
    return df

# ランクラベルに対応する利得（DCG用）
def make_label_gain(n_bins=18, mode='sqrt'):
    if mode == 'linear':
        return list(range(0, n_bins+1))
    if mode == 'sqrt':
        return [0] + [np.sqrt(i) for i in range(1, n_bins+1)]
    if mode == 'exp':
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    if mode == 'dcg':
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    return list(range(0, n_bins+1))

# Target Encoding（平滑化付き）
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

# レース単位にグループ化 → Tensor形式に（単勝用）
def group_by_race(df_part):
    X_groups, y_groups, win_groups, payout_groups = [], [], [], []
    cat_groups, context_num_groups, context_cat_groups = [], [], []
    win_index_groups = []

    for _, g in df_part.groupby(cfg.group_col):
        X = g[cfg.feature_cols].values.astype(np.float32)
        y = g[cfg.target_col].values.astype(np.float32)
        is_win = g["is_win"].values.astype(np.float32)
        payout = g["オッズ"].values.astype(np.float32) - 1.0
        cat_X = g[cfg.embedding_cols].values.astype(np.int64)
        context_num = g[cfg.context_num_cols].iloc[0].values.astype(np.float32)
        context_cat = g[cfg.context_cat_cols].iloc[0].values.astype(np.int64)
        num_horses = len(g)
        context_num = np.tile(context_num, (num_horses, 1))
        context_cat = np.tile(context_cat, (num_horses, 1))

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        win_t = torch.tensor(is_win, dtype=torch.float32)
        payout_t = torch.tensor(payout, dtype=torch.float32)
        cat_t = torch.tensor(cat_X, dtype=torch.long)
        context_num_t = torch.tensor(context_num, dtype=torch.float32)
        context_cat_t = torch.tensor(context_cat, dtype=torch.long)

        if win_t.sum() == 1:
            win_idx = torch.argmax(win_t).long()
        else:
            win_idx = torch.tensor(-1).long()
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

# レース単位にグループ化 → Tensor形式に（複勝用）
def group_by_race_fuku(df_part):
    X_groups, y_groups, win_groups, payout_groups = [], [], [], []
    cat_groups, context_num_groups, context_cat_groups = [], [], []
    win_index_groups = []

    for _, g in df_part.groupby(cfg.group_col):
        X = g[cfg.feature_cols].values.astype(np.float32)
        y = g[cfg.target_col].values.astype(np.float32)
        is_win = g["複勝_hit"].values.astype(np.float32)
        payout = (g["複勝払戻"].values.astype(np.float32) / 100)
        cat_X = g[cfg.embedding_cols].values.astype(np.int64)
        context_num = g[cfg.context_num_cols].iloc[0].values.astype(np.float32)
        context_cat = g[cfg.context_cat_cols].iloc[0].values.astype(np.int64)
        num_horses = len(g)
        context_num = np.tile(context_num, (num_horses, 1))
        context_cat = np.tile(context_cat, (num_horses, 1))

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        win_t = torch.tensor(is_win, dtype=torch.float32)
        payout_t = torch.tensor(payout, dtype=torch.float32)
        cat_t = torch.tensor(cat_X, dtype=torch.long)
        context_num_t = torch.tensor(context_num, dtype=torch.float32)
        context_cat_t = torch.tensor(context_cat, dtype=torch.long)

        if win_t.sum() == 1:
            win_idx = torch.argmax(win_t).long()
        else:
            win_idx = torch.tensor(-1).long()
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

# 重み初期化（kaiming / xavier）
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)

# カテゴリ特徴一覧を返す
def embedding_init():
    emb = cfg.feature_category
    return emb

# シード固定＋DataLoader用seed_worker
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker
