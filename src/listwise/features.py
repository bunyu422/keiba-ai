import copy
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from src.listwise import model_config as cfg
from src.common import transform
from src.common.transform import (
    softmax, softmax_neg_rank, make_smooth_relevance_labels,
    make_rank_labels, make_label_gain, init_weights, set_seed,
    pick_top_random, fill_nan,
)

# CSV読み込み
def load_csv(path):
    df = pd.read_csv(path, index_col=0)
    return df

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
    return transform.race_feature(df, cfg.feature_category, cfg.context_cat_cols)

# カテゴリカラム変換（学習用・マッピング保存）
def race_feature_train(df):
    return transform.race_feature_train(df, cfg.feature_category, cfg.context_cat_cols)

# カテゴリカラム変換（テスト用・学習時のマッピングを参照）
def race_feature_test(df, category_mappings):
    return transform.race_feature_test(df, category_mappings, cfg.feature_category, cfg.context_cat_cols)

# 反転カラムを符号反転
def inversion(df):
    return transform.inversion(df, cfg.inversion_cols)

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
        y = g['smooth_rel'].values.astype(np.float32)
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

# カテゴリ特徴一覧を返す
def embedding_init():
    emb = cfg.feature_category
    return emb

def add_race_relative_features(
    df, cols, group_col="レースID",
    add_rank=True, add_relative=True, add_zscore=True,
):
    df = df.copy()
    col_list = []
    rank_cols = {}
    grouped = df.groupby(group_col)
    means = grouped[cols].transform('mean')
    stds = grouped[cols].transform('std').replace(0, np.nan)

    for col in cols:
        if add_rank:
            rank_cols[f"{col}_rank"] = grouped[col].rank(ascending=False, method="dense")
            col_list.append(f"{col}_rank")
        if add_relative:
            rank_cols[f"{col}_rel"] = df[col] - means[col]
            col_list.append(f"{col}_rel")
        if add_zscore:
            rank_cols[f"{col}_z"] = (df[col] - means[col]) / stds[col]
            col_list.append(f"{col}_z")

    df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)
    return df, col_list


def add_race_condition_scores(
    df, group_distance_col="距離", group_field_col="フィールド",
    group_baba_col="馬場", n_past=5, score_col="スピード指数",
):
    df = df.copy()

    dist_scores = []
    for i in range(1, n_past + 1):
        diff = (df[f"{i}距離"] - df[group_distance_col]).abs()
        s = df[f"{i}{score_col}"] / (1 + diff / 200)
        dist_scores.append(s)
    df["距離適性スコア"] = np.vstack(dist_scores).mean(axis=0)

    baba_scores = []
    for i in range(1, n_past + 1):
        same = (df[f"{i}馬場"] == df[group_baba_col]).astype(float)
        s = df[f"{i}{score_col}"] * (0.5 + 0.5 * same)
        baba_scores.append(s)
    df["馬場適性スコア"] = np.vstack(baba_scores).mean(axis=0)

    field_scores = []
    for i in range(1, n_past + 1):
        same = (df[f"{i}フィールド"] == df[group_field_col]).astype(float)
        s = df[f"{i}{score_col}"] * (0.5 + 0.5 * same)
        field_scores.append(s)
    df["フィールド適性スコア"] = np.vstack(field_scores).mean(axis=0)

    return df


def build_winner_baseline(df, field, n_past=5):
    past_rows = []
    for n in range(1, n_past + 1):
        cols = {
            '場所_base': f'{n}場所', '距離_base': f'{n}距離',
            'フィールド_base': f'{n}フィールド', 'クラス_base': f'{n}クラス',
            '馬場_base': f'{n}馬場', '過去着順_base': f'{n}過去着順',
            '後3F_base': f'{n}後3F', 'タイム_base': f'{n}タイム',
            'スピード指数_base': f'{n}スピード指数', '馬体重_base': f'{n}馬体重',
            '馬番_base': f'{n}馬番', 'コーナー通過順_base': f'{n}コーナー通過順',
            '斤量_base': f'{n}斤量',
        }
        tmp = df[[v for v in cols.values()]].copy()
        tmp.rename(columns={v: k for k, v in cols.items()}, inplace=True)
        tmp["走前"] = n
        tmp = tmp[tmp["過去着順_base"] == 1]
        past_rows.append(tmp)

    past_df = pd.concat(past_rows, ignore_index=True)
    baseline = (
        past_df
        .groupby(["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"])
        [["後3F_base", "タイム_base", "スピード指数_base", "馬体重_base",
          "コーナー通過順_base", "馬番_base", '斤量_base']]
        .mean().reset_index()
    )
    baseline.to_csv(f"./csv/{field}_winner_baseline.csv", index=False)
    return baseline


def add_past_diff_features(df, baseline, n_past=5):
    df2 = df.copy()
    diff_cols = []
    score_cols = []
    convert_cols = []

    for n in range(1, n_past + 1):
        convert_cols.extend([f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"])

    df2[convert_cols] = df2[convert_cols].astype(float)

    for n in range(1, n_past + 1):
        key_cols = [f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"]
        merged = df2.merge(
            baseline,
            left_on=key_cols,
            right_on=["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"],
            how="left", suffixes=("", "_base")
        ).reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        feature_list = ["後3F", "タイム", "スピード指数", "馬体重", "コーナー通過順", "馬番", "斤量"]
        score_items = []
        for col in feature_list:
            diff_col = f"{n}{col}_diff"
            df2[diff_col] = merged[f"{n}{col}"] - merged[f"{col}_base"]
            diff_cols.append(diff_col)
            diff_val = df2[diff_col]
            if col in ["後3F", "タイム"]:
                score = -diff_val
            elif col in ["スピード指数", "斤量"]:
                score = diff_val
            elif col in ["馬体重", "コーナー通過順", "馬番"]:
                score = diff_val.abs()
            else:
                score = 0
            score_items.append(score)

        score_col = f"{n}_past_score"
        df2[score_col] = sum(score_items)
        score_cols.append(score_col)

    score_df = df2[score_cols]
    df2["past_score_mean"] = score_df.mean(axis=1)
    df2["past_score_max"] = score_df.max(axis=1)
    df2["past_score_min"] = score_df.min(axis=1)
    df2["past_score_sum"] = score_df.sum(axis=1)

    alpha = 0.7
    weights = np.array([np.exp(-alpha * i) for i in range(n_past)])
    weights = weights / weights.sum()
    df2["past_score_ewm"] = (score_df.values * weights).sum(axis=1)

    score_cols.extend(["past_score_mean", "past_score_max", "past_score_min", "past_score_sum", "past_score_ewm"])
    return df2, diff_cols, score_cols


def add_fuku_payout(df_pred, df_payout, race_col="レースID"):
    fuku = df_payout.query("券種 == '複勝'")[[race_col, "馬番", "払い戻し金額"]].copy()
    fuku["馬番"] = fuku["馬番"].astype(float).astype(str)
    df_pred["馬番"] = df_pred["馬番"].astype(str)
    merged = df_pred.merge(fuku, on=[race_col, "馬番"], how="left")
    merged["複勝_hit"] = merged["払い戻し金額"].notna().astype(int)
    merged["複勝払戻"] = merged["払い戻し金額"].fillna(0)
    return merged


# ============================================================ #
# 新規6特徴量
# ============================================================ #

# --- 距離グループマッピング ---
def _distance_group(dist):
    if dist <= 1400:
        return 0  # 短距離
    elif dist <= 1800:
        return 1  # マイル
    elif dist <= 2400:
        return 2  # 中距離
    else:
        return 3  # 長距離

# --- Feature 1~3: 交互作用TE (fold内で使用) ---
def add_interaction_te_fold(train_df, val_df, test_df, df_2025, group_cols, target='着順', alpha=20):
    """fold内で interaction TE を計算（内部5-fold CV）し train/val/test に適用"""
    col_name = '_'.join(group_cols) + '_te'
    global_mean = train_df[target].mean()
    train_df = train_df.copy()

    # 内部5-fold CVで train の TE を計算（リーク防止）
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_df[col_name] = np.nan
    for train_cv_idx, val_cv_idx in kf.split(train_df):
        cv_train = train_df.iloc[train_cv_idx]
        cv_val = train_df.iloc[val_cv_idx]
        stats = cv_train.groupby(group_cols)[target].agg(['mean', 'count'])
        stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)
        cv_mapping = stats['smooth'].to_dict()
        def _cv_map(row):
            return cv_mapping.get(tuple(row[g] for g in group_cols), global_mean)
        train_df.iloc[val_cv_idx, train_df.columns.get_loc(col_name)] = cv_val.apply(_cv_map, axis=1).values

    # 全 train からマッピングを構築 → val/test に適用
    stats = train_df.groupby(group_cols)[target].agg(['mean', 'count'])
    stats['smooth'] = (stats['mean'] * stats['count'] + alpha * global_mean) / (stats['count'] + alpha)
    mapping = stats['smooth'].to_dict()
    def _map_row(row):
        return mapping.get(tuple(row[g] for g in group_cols), global_mean)

    val_df = val_df.copy()
    test_df = test_df.copy()
    df_2025 = df_2025.copy()
    val_df[col_name] = val_df.apply(_map_row, axis=1)
    test_df[col_name] = test_df.apply(_map_row, axis=1)
    df_2025[col_name] = df_2025.apply(_map_row, axis=1)

    return train_df, val_df, test_df, df_2025, col_name, mapping


# --- Feature 4: 出走間隔クラス ---
def add_interval_class(df):
    """出走間隔（週数）をカテゴリ化"""
    df = df.copy()
    def _classify(x):
        if pd.isna(x) or x == 0:
            return 0  # 連闘
        elif x <= 2:
            return 1  # 中1週
        elif x <= 4:
            return 2  # 中2週
        elif x <= 8:
            return 3  # 休み明け
        else:
            return 4  # 長期休養
    df['間隔クラス'] = df['間隔'].apply(_classify)
    return df


# --- Feature 5: 上がり3F レース内ランク ---
def add_last3f_race_rank(df):
    """各馬の前走の上がり3F（1後3F）をレース内で順位付け"""
    df = df.copy()
    df['前走後3F_レース内順位'] = df.groupby('レースID')['1後3F'].rank(ascending=True, method='dense')
    return df


# --- Feature 6: 馬体重トレンド傾き ---
def add_weight_trend_slope(df, n_past=5):
    """過去 n_past 走の馬体重の線形回帰傾き"""
    df = df.copy()
    weight_cols = [f'{i}馬体重' for i in range(1, n_past + 1)]
    def _calc_slope(row):
        weights = row[weight_cols].values.astype(float)
        mask = ~np.isnan(weights)
        if mask.sum() < 2:
            return 0.0
        x = np.arange(n_past)[mask]
        slope, _ = np.polyfit(x, weights[mask], 1)
        return slope
    df['馬体重_trend_slope'] = df.apply(_calc_slope, axis=1)
    return df


# --- 距離グループ列を追加 ---
def add_distance_group(df):
    """距離を4グループに分類"""
    df = df.copy()
    df['距離グループ'] = df['距離'].apply(_distance_group)
    return df


def add_common_cols_extended():
    """common_cols に距離グループを追加"""
    if '距離グループ' not in cfg.common_cols:
        cfg.common_cols.append('距離グループ')
    if '間隔クラス' not in cfg.common_cols:
        cfg.common_cols.append('間隔クラス')


def add_fuku_payout_maxonly(df_pred, df_payout, race_col="レースID"):
    fuku = df_payout.query("券種 == '複勝'")[[race_col, "馬番", "払い戻し金額"]].copy()
    fuku["馬番"] = fuku["馬番"].astype(float).astype(str)
    df_pred["馬番"] = df_pred["馬番"].astype(str)
    merged = df_pred.merge(fuku, on=[race_col, "馬番"], how="left")
    merged["複勝_hit"] = merged["払い戻し金額"].notna().astype(int)
    merged["複勝払戻"] = merged["払い戻し金額"].fillna(0)
    merged["複勝_hit_max"] = 0
    merged["複勝払戻_max"] = 0

    results = []
    for race_id, g in merged.groupby(race_col):
        max_pay = g["払い戻し金額"].max()
        if pd.isna(max_pay) or max_pay == 0:
            results.append(g)
            continue
        candidates = g[g["払い戻し金額"] == max_pay]
        best = candidates.sort_values("人気", na_position="last").iloc[0]
        g2 = g.copy()
        g2.loc[best.name, "複勝_hit_max"] = 1
        g2.loc[best.name, "複勝払戻_max"] = max_pay
        results.append(g2)

    merged = pd.concat(results).sort_index()
    return merged
