"""
特徴量変換ユーティリティ

inversion / fill_nan / race_feature / softmax / set_seed 等の
LightGBM・ListNet 両方から利用される共通機能を集約。
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def inversion(df, inversion_cols):
    """inversion_cols の符号を反転する（小さいほど良い値を正にする）。"""
    df = df.copy()
    for col in inversion_cols:
        df[col] = -df[col]
    return df


def fill_nan(df, cols, fill_value=-9999):
    """指定カラムの欠損値を fill_value で埋める。"""
    df[cols] = df[cols].fillna(fill_value)
    return df


def race_feature(df, category, context_cat):
    """カテゴリカラムを category 型→コード変換（学習用・コード維持）。"""
    df = df.copy()
    all_cats = category + context_cat
    for col in all_cats:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes
    return df


def race_feature_train(df, category, context_cat):
    """カテゴリカラム変換（学習用・マッピング保存）。"""
    df = df.copy()
    category_mappings = {}
    all_cats = category + context_cat
    for col in all_cats:
        df[col] = df[col].astype('category')
        category_mappings[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes
    return df, category_mappings


def race_feature_test(df, category_mappings, category, context_cat):
    """カテゴリカラム変換（テスト用・学習時のマッピングを参照）。"""
    df = df.copy()
    all_cats = category + context_cat
    for col in all_cats:
        if col in df.columns:
            df[col] = df[col].astype(str)
            inv_map = {v: k for k, v in category_mappings[col].items()}
            key_type = type(list(inv_map.keys())[0])
            n_train_categories = len(category_mappings[col])
            df[col] = df[col].map(
                lambda x: inv_map.get(key_type(x), n_train_categories)
            ).astype(int)
    return df


def softmax(x):
    """通常の softmax。"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def softmax_neg_rank(rankings):
    """順位（小さいほど良い）を反転した softmax 確率。"""
    x = -rankings
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def make_smooth_relevance_labels(df, max_rel=3.0):
    """滑らかな関連度ラベル: 着順に応じて [0, max_rel]。"""
    return (max_rel / np.log1p(df['着順'] + 1)).clip(0, max_rel)


def make_rank_labels(df, group_col='レースID', pos_col='着順', n_bins=18):
    """着順をレース内相対ランクに変換し bin 分割で離散化。"""
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


def make_top2_labels(df, pos_col='着順'):
    """3-level label for wide optimization: 1st=2, 2nd=1, else=0"""
    df = df.copy()
    df['rank_label'] = df[pos_col].map(lambda x: 2 if x == 1 else (1 if x == 2 else 0))
    return df


def make_label_gain(n_bins=18, mode='sqrt'):
    """ランクラベルに対応する利得（DCG 用）。"""
    if mode == 'linear':
        return list(range(0, n_bins + 1))
    if mode == 'sqrt':
        return [0] + [np.sqrt(i) for i in range(1, n_bins + 1)]
    if mode == 'exp':
        return [0] + [2**i - 1 for i in range(1, n_bins + 1)]
    if mode == 'dcg':
        return [0] + [2**i - 1 for i in range(1, n_bins + 1)]
    return list(range(0, n_bins + 1))


def set_seed(seed=42):
    """シード固定＋DataLoader 用 seed_worker。"""
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


def init_weights(m):
    """重み初期化（kaiming / xavier）。"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)


def pick_top_random(df, score_col="pred_score", race_col="レースID"):
    """レースごとにスコア最大の馬をランダムに 1 頭選択。"""
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
