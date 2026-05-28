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
from tqdm import tqdm
import Learning
import torch.nn.functional as F
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
from sklearn.model_selection import StratifiedGroupKFold
import Listwise as lw
import seaborn as sns
from scipy.stats import entropy, wasserstein_distance
from sklearn.ensemble import ExtraTreesClassifier


# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)
# 小数点をすべて表示（指数表記なし）
pd.set_option('display.float_format', lambda x: f'{x:.16f}'.rstrip('0').rstrip('.'))
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def load_csv(path):
    # 学習データを読み込む
    df = pd.read_csv(path, index_col=0)
    return df

def split_test_for_next_stage(test_df, group_col="レースID", ratios=(0.3, 0.1, 0.6)):
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

def oof_ridge(target_col, train_df, val_df, test_df, df_2025, feature_cols, name):
    oof_pred = np.zeros(len(train_df))
    gkf = GroupKFold(n_splits=5)
    not_pop = [c for c in feature_cols if c != target_col]
    joblib.dump(not_pop, f"./model/clf_{name}_input_{field}.pkl")

    for tr_idx, va_idx in gkf.split(train_df, groups=train_df['レースID']):
        X = train_df.iloc[tr_idx][not_pop]
        y = train_df.iloc[tr_idx][target_col]
        mask = ~y.isna()
        imputer = SimpleImputer(strategy='median')
        clf = make_pipeline(imputer, Ridge(alpha=1.0))
        clf.fit(X[mask], y[mask])
        oof_pred[va_idx] = clf.predict(train_df.iloc[va_idx][not_pop])

    # train の残差（OOF）
    train_df[target_col] = train_df[target_col].astype(float) - oof_pred

    # valid/test は train で fit したモデルの全体版を使って予測
    X = train_df[not_pop]
    y = train_df[target_col]
    mask = ~y.isna()
    imputer = SimpleImputer(strategy='median')
    clf = make_pipeline(imputer, Ridge(alpha=1.0))
    clf.fit(X[mask], y[mask])
    joblib.dump(clf, f"./model/clf_{name}_model_{field}_fold{fold}.pkl")


    pred_pop = clf.predict(val_df[not_pop])
    val_df[target_col] = val_df[target_col].astype(float) - pred_pop

    pred_pop = clf.predict(test_df[not_pop])
    test_df[target_col] = test_df[target_col].astype(float) - pred_pop

    pred_pop = clf.predict(df_2025[not_pop])
    df_2025[target_col] = df_2025[target_col].astype(float) - pred_pop

    return train_df, val_df, test_df, df_2025


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
    return winrate_mapping, combo_list
    
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
        for stat_key, jp in zip(["win_rate", "place_rate", "show_rate", "n"], ["勝率", "連対率", "複勝率", "件数"]):
            col_name = f"{feature}_{jp}"
            df[col_name] = df.apply(
                lambda r: get_stat(
                    r["コース距離"],
                    feature,
                    tuple(r[c] for c in feature.split('_')),  # ここでスカラー値をタプル化
                    jp
                ),
                axis=1
            )

    return df

def select_similar_train_races_at_k(
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
    df["kaisaikai"] = df[group_col].astype(str).str[6:8].astype(int)
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

# def apply_conditions(df, X, model, conditions):
#     hits = np.zeros(len(df), dtype=bool)

#     for _, row in conditions.iterrows():
#         tree = model.estimators_[row.tree_id]
#         leaf = tree.apply(X)
#         hits |= (leaf == row.leaf)

#     return hits

def apply_conditions(df, X, model, valid_conditions):
    hit_scores = np.zeros(len(df))

    for _, row in valid_conditions.iterrows():
        tree = model.estimators_[row.tree_id]
        leaf = tree.apply(X)
        hit_scores += (leaf == row.leaf)

    df = df.copy()
    df["hit_score"] = hit_scores

    # レース内で hit_score 最大の1頭だけ
    idx = (
        df[df.hit_score > 0]
        .groupby("レースID")["hit_score"]
        .idxmax()
    )

    mask = pd.Series(False, index=df.index)
    mask.loc[idx] = True
    return mask

def apply_conditions_weight(
    df,
    X,
    model,
    summary_df,
    min_score=1   # ← 1以上ヒットしたら賭ける
):
    hit_scores = np.zeros(len(df))

    for _, row in summary_df.iterrows():
        tree_id = row.tree
        leaf_id = row.leaf

        leaf = model.estimators_[tree_id].apply(X)
        hit_scores += (leaf == leaf_id).astype(int)

    df = df.copy()
    df["hit_score"] = hit_scores

    # レース内で hit_score 最大 & min_score 以上
    idx = (
        df[df.hit_score >= min_score]
        .groupby("レースID")["hit_score"]
        .idxmax()
    )

    mask = pd.Series(False, index=df.index)
    mask.loc[idx] = True
    return mask

def get_leaf_conditions(tree, feature_names, leaf_id):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    paths = []

    def recurse(node, path):
        if node == leaf_id:
            paths.append(path)
            return
        if tree_.children_left[node] != -1:
            f = feature[node]
            thr = threshold[node]
            recurse(tree_.children_left[node],
                    path + [(f, "<=", thr)])
            recurse(tree_.children_right[node],
                    path + [(f, ">", thr)])

    recurse(0, [])
    
    # feature index → name
    readable = []
    for path in paths:
        readable.append([
            f"{feature_names[f]} {op} {thr:.3f}"
            for f, op, thr in path
        ])
    return readable

def count_hits(df, X, model, tree_id, leaf_id):
    leaf = model.estimators_[tree_id].apply(X)
    mask = (leaf == leaf_id)
    return mask.sum(), mask

def calc_roi(df, mask):
    return df.loc[mask, "単勝オッズ"].sum() / (100 * mask.sum())

def compute_score(df, X, model, summary_df):
    score = np.zeros(len(df))

    for _, row in summary_df.iterrows():
        tree_id = int(row.tree)   # ← ★ここ
        leaf_id = int(row.leaf)   # ← ★念のため

        w = row.weight

        leaf = model.estimators_[tree_id].apply(X)
        score += (leaf == leaf_id) * w

    return score

# file_path
###########################モデルごとに変更が必要############################
ninki = True
field = 'chukyo'
csv_path = f'./csv/df_all_{field}_2025_add.csv'
model_type = "reg-to-rank"
# csv_path = f'./csv/df_all_{field}.csv'
###########################################################################

df = load_csv(csv_path)
group_col = 'レースID'
target_col = 'is_win'
feature_cols = []
df['is_win'] = (df['着順'] == 1).astype(int)

candidate_cols = ['齢', '間隔', '父馬', '騎手', '性', '齢']

cat_list = ['距離', 'フィールド', '馬場', '父馬', '騎手', '性']

if __name__ == '__main__':
    # print(len(df))
    # print(df['レースID'].unique().tolist())
    seed = 1
    lw.set_seed(seed)  # 先に乱数固定
    fold_results = []
    print(df.columns.values)

    # === 5. KFold処理 ===
    
    # ラベル作成
    # df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())
    # df['人気'] = df['人気'].fillna(df['人気'].median())
    # df['smooth_rel'] = lw.make_smooth_relevance_labels(df)
    # df = make_rank_labels(df)

    # 出走頭数ビン
    # bins_horses = [0, 13, 16, 100]
    # labels_horses = ['small', 'medium', 'large']
    # df['num_horses_bin'] = pd.cut(df['出走頭数'], bins=bins_horses, labels=labels_horses)

    # 反転
    # df = lw.inversion(df)

    # # カラム追加
    # df = lw.append_col(df)
    # df = lw.add_relative_features(df)

    # df['label'] = 0
    # df.loc[df['人気'].astype(int)==-1, 'label'] = 5
    # df.loc[df['人気'].astype(int)==-2, 'label'] = 4
    # df.loc[df['人気'].astype(int)==-3, 'label'] = 3
    # df.loc[df['人気'].astype(int)==-1, 'label'] = 2
    # df.loc[df['人気'].astype(int)==-2, 'label'] = 1
    
    # print(df['label'].value_counts())
    # df[['1クラス', '2クラス', '3クラス', '4クラス', '5クラス']] = df[['1クラス', '2クラス', '3クラス', '4クラス', '5クラス']].astype(int)


    # splits, df_test_2025, df = time_series_group_cv_3split_2025(df)
    splits, df = time_series_group_cv_3split_2025(df)
    combo_list = None
    year = 2025

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        # if fold == 1:
        #     break

        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]
        print("fold_num:", len(train_df))
        print("fold_num:", len(val_df))
        print("fold_num:", len(test_df))
        # print("fold_num:", len(df_test_2025))
        year = year - 1
        # train_df, val_df, test_df = time_series_group_split_by_year(df, year=year)

        # === 6. 特徴量エンコーディング ===
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        # test_df  = df_test_2025.reset_index(drop=True)
        df_2025  = test_df.reset_index(drop=True)

        # winrate_mapping, combo_list = generate_stat_features_by_course_train(train_df, candidate_cols, n_sample=10, combo_list=combo_list)
        # train_df = apply_stats_from_mapping(train_df, winrate_mapping)
        # val_df = apply_stats_from_mapping(val_df, winrate_mapping)
        # test_df = apply_stats_from_mapping(test_df, winrate_mapping)
        # df_2025 = apply_stats_from_mapping(df_2025, winrate_mapping)

        train_df, sire_mapping = lw.target_encoding(train_df, '父馬', target_col)
        # with open(f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl', "wb") as dd:
        #     pickle.dump(sire_mapping, dd)

        # val/test は train 全体の mapping を使う
        val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
        test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)
        df_2025['父馬_te'] = df_2025['父馬'].map(sire_mapping).fillna(-1)

        train_df, j_mapping = lw.target_encoding(train_df, '騎手', target_col)
        # with open(f'./pickle-dict/jwin_dict_{field}_fold{fold}.pkl', "wb") as dd:
        #     pickle.dump(j_mapping, dd)

        # val/test は train 全体の mapping を使う
        val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
        test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)
        df_2025['騎手_te'] = df_2025['騎手'].map(j_mapping).fillna(-1)
                
        feature_cols = [col for col in train_df.columns if col not in ["払い戻し金額", "複勝払戻", "複勝_hit", "is_fuku", "is_ren", 'コース距離', 'label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
        # train_df, temp = select_similar_train_races_at_k(pd.concat([train_df, val_df], axis=0, ignore_index=True), feature_cols=feature_cols, target_year=year, top_k=5)

        train_df, val_df, test_df, df_2025 = lw.fill_nan(train_df, feature_cols), lw.fill_nan(val_df, feature_cols), lw.fill_nan(test_df, feature_cols), lw.fill_nan(df_2025, feature_cols)

        # zero_var = train_df[scale_cols].std()[train_df[scale_cols].std() == 0]
        # print("分散ゼロの列:", zero_var.index.tolist())

        # print("train shape:", train_df[scale_cols].shape)
        # print("val shape:", val_df[scale_cols].shape)
        # print("test shape:", test_df[scale_cols].shape)

        # print("NaN 含有数:\n", train_df[scale_cols].isna().sum())
        # print("有効サンプル数:", train_df[scale_cols].notna().sum())

        # === 7. 特徴量スケーリング ===
        # scaler = StandardScaler()
        # train_df[lw.scale_cols] = scaler.fit_transform(train_df[lw.scale_cols])
        # val_df[lw.scale_cols] = scaler.transform(val_df[lw.scale_cols])
        # test_df[lw.scale_cols] = scaler.transform(test_df[lw.scale_cols])
        # df_2025[lw.scale_cols] = scaler.transform(df_2025[lw.scale_cols])

        # # スケーラーを保存（モデルと同じディレクトリに置くのが一般的）
        # joblib.dump(scaler, f"./model/scaler_{field}_fold{fold}.pkl")

        # === 0. データの前処理 ===
        # Nanの処理
        # train_df, val_df, test_df, df_2025 = lw.fill_nan(train_df, feature_cols), lw.fill_nan(val_df, feature_cols), lw.fill_nan(test_df, feature_cols), lw.fill_nan(df_2025, feature_cols)
        # カテゴリ変換
        
        # train_df, map_dict = lw.race_feature_train(train_df)
        # val_df = lw.race_feature_test(val_df, map_dict)
        # test_df = lw.race_feature_test(test_df, map_dict)
        # df_2025 = lw.race_feature_test(df_2025, map_dict)

        # train_df = train_df.round(10)
        # val_df = val_df.round(10)
        # test_df = test_df.round(10)
        # df_2025 = df_2025.round(10)

        # train_df, val_df, test_df, df_2025 = oof_ridge('人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki")
        # train_df, val_df, test_df, df_2025 = oof_ridge('1人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki1")
        # train_df, val_df, test_df, df_2025 = oof_ridge('2人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki2")
        # train_df, val_df, test_df, df_2025 = oof_ridge('3人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki3")
        # train_df, val_df, test_df, df_2025 = oof_ridge('4人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki4")
        # train_df, val_df, test_df, df_2025 = oof_ridge('5人気', train_df, val_df, test_df, df_2025, feature_cols, "ninki5")

        # 保存
        # joblib.dump(map_dict, f"./pickle-dict/category_mappings_{field}_fold{fold}.pkl")
        # joblib.dump(feature_cols, f"./pickle-dict/{field}_lgb_cols.pkl")

        # print(val_df.head(30))

        bad_vals = ~np.isfinite(train_df.select_dtypes(include=[np.number]))

        X_train = train_df[feature_cols].values
        X_val   = val_df[feature_cols].values
        X_test  = test_df[feature_cols].values

        model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=100,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, train_df[target_col])

        leaf_ids = []

        for tree in model.estimators_:
            leaf_ids.append(tree.apply(X_train))

        leaf_ids = np.vstack(leaf_ids).T  # [n_samples, n_trees]

        train_df = train_df.copy()

        roi_records = []

        for t_idx in range(leaf_ids.shape[1]):
            train_df["leaf"] = leaf_ids[:, t_idx]

            g = train_df.groupby("leaf").agg(
                n=("leaf", "size"),
                roi=("単勝オッズ", lambda x: x.sum() / (100 * len(x)))
            )

            g["tree_id"] = t_idx
            roi_records.append(g.reset_index())

        roi_df = pd.concat(roi_records, ignore_index=True)

        valid_conditions = roi_df[
            (roi_df["n"] >= 150) &
            (roi_df["roi"] > 1.05)
        ][["tree_id", "leaf"]]

        # summary
        # records = []

        # for _, row in valid_conditions.iterrows():
        #     tree_id = row.tree_id
        #     leaf_id = row.leaf

        #     leaf = model.estimators_[tree_id].apply(X_val)
        #     mask = (leaf == leaf_id)

        #     n_val = mask.sum()
        #     if n_val == 0:
        #         continue

        #     roi_val = val_df.loc[mask, "単勝オッズ"].sum() / (100 * n_val)

        #     records.append({
        #         "tree": tree_id,
        #         "leaf": leaf_id,
        #         "n_val": n_val,
        #         "roi_val": roi_val
        #     })

        # summary_df = pd.DataFrame(records)

        # # ★ VALでは「条件選別」だけ
        # summary_df = summary_df[
        #     (summary_df["n_val"] >= 50) &
        #     (summary_df["roi_val"] > 1.05)   # 少し厳しめ
        # ]

        # summary_df["weight"] = np.log(summary_df["roi_val"])

        # test_df = test_df.copy()
        # test_df["score"] = compute_score(test_df, X_test, model, summary_df)
        # candidates = test_df[test_df['score'] > 0]

        # idx = (
        #     candidates
        #     .groupby("レースID")["score"]
        #     .idxmax()
        # )

        # mask = pd.Series(False, index=test_df.index)
        # mask.loc[idx] = True

        # roi = (
        #     test_df.loc[mask, "単勝オッズ"].sum()
        #     / (100 * mask.sum())
        # )

        # print("TEST ROI:", roi)
        # print("bets:", mask.sum())

        # for thr in [0.0, 0.2, 0.5, 1.0, 1.5]:
        #     idx = (
        #         test_df[test_df.score >= thr]
        #         .groupby("レースID")["score"]
        #         .idxmax()
        #     )

        #     if len(idx) == 0:
        #         continue

        #     roi = (
        #         test_df.loc[idx, "単勝オッズ"].sum()
        #         / (100 * len(idx))
        #     )

        #     print(f"thr={thr:.2f}, bets={len(idx)}, ROI={roi:.3f}")

        # test_mask = apply_conditions_weight(
        #     test_df,
        #     X_test,
        #     model,
        #     summary_df,
        #     min_score=0.0
        # )

        # test_roi = (
        #     test_df.loc[test_mask, "単勝オッズ"].sum()
        #     / (100 * test_mask.sum())
        # )

        # print("TEST ROI:", test_roi)
        # print("bets:", test_mask.sum())
        # print("avg odds:", test_df.loc[test_mask, "単勝オッズ"].mean())

        # val
        # val_hits = apply_conditions(val_df, X_val, model, valid_conditions)

        # val_roi = val_df.loc[val_hits, "単勝オッズ"].sum() / (100 * val_hits.sum())
        # print("VAL ROI:", val_roi)

        # # test
        # test_hits = apply_conditions(test_df, X_test, model, valid_conditions)

        # test_roi = test_df.loc[test_hits, "単勝オッズ"].sum() / (100 * test_hits.sum())
        # print("TEST ROI:", test_roi)
        

        # # 条件確認
        # row = valid_conditions.iloc[0]
        # tree_id = row["tree_id"]
        # leaf_id = row["leaf"]

        # conds = get_leaf_conditions(
        #     model.estimators_[tree_id],
        #     feature_cols,
        #     leaf_id
        # )

        # for c in conds:
        #     print(" AND ".join(c))

        # n_train, train_mask = count_hits(
        #     train_df, X_train, model, tree_id, leaf_id
        # )
        # n_val, val_mask = count_hits(
        #     val_df, X_val, model, tree_id, leaf_id
        # )
        # n_test, test_mask = count_hits(
        #     test_df, X_test, model, tree_id, leaf_id
        # )

        # print(n_train, n_val, n_test)

        # print("train:", calc_roi(train_df, train_mask))
        # print("val  :", calc_roi(val_df, val_mask))
        # print("test :", calc_roi(test_df, test_mask))

        summary = []

        for _, row in valid_conditions.iterrows():
            tree_id = row.tree_id
            leaf_id = row.leaf

            n_tr, tr_mask = count_hits(train_df, X_train, model, tree_id, leaf_id)
            n_va, va_mask = count_hits(val_df, X_val, model, tree_id, leaf_id)
            n_te, te_mask = count_hits(test_df, X_test, model, tree_id, leaf_id)

            summary.append({
                "tree": tree_id,
                "leaf": leaf_id,
                "n_train": n_tr,
                "roi_train": calc_roi(train_df, tr_mask),
                "n_val": n_va,
                "roi_val": calc_roi(val_df, va_mask),
                "n_test": n_te,
                "roi_test": calc_roi(test_df, te_mask),
            })

        summary_df = pd.DataFrame(summary).sort_values("roi_val", ascending=False)

        print(summary_df)

        pass
        # test_df.to_csv(f'./csv/{field}_result_lgb_test_{fold}.csv', index=False)
        # df_2025.to_csv(f'./csv/{field}_result_lgb_2025_{fold}.csv', index=False)