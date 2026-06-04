import itertools
import pickle
import random
import numpy as np
import pandas as pd
from scipy.stats import entropy, wasserstein_distance
from tqdm import tqdm

# 時系列CV: レースID順に train/val/test に分割（ランダム比率）
def time_series_group_cv_3split_2025(df, group_col="レースID", n_folds=5):
    df_rest = df
    unique_races = np.sort(df_rest[group_col].unique())
    n_races = len(unique_races)
    stage_splits = []

    for i in range(n_folds):
        test_ratio = random.uniform(0.1, 0.3)
        val_ratio = 0.1
        train_ratio = 1 - (test_ratio + val_ratio)
        train_end = int(n_races * train_ratio)
        val_end = int(n_races * (train_ratio + val_ratio))
        train_races = unique_races[:train_end]
        val_races = unique_races[train_end:val_end]
        test_races = unique_races[val_end:]
        train_idx = df_rest[df_rest[group_col].isin(train_races)].index
        val_idx = df_rest[df_rest[group_col].isin(val_races)].index
        test_idx = df_rest[df_rest[group_col].isin(test_races)].index
        stage_splits.append((train_idx, val_idx, test_idx))
    return stage_splits, df_rest

# 単純な時系列分割: 先頭train_ratio%をtrain, 残りをval/test
def time_series_train_val_split(df, group_col="レースID", train_ratio=0.8):
    unique_races = np.sort(df[group_col].unique())
    n_races = len(unique_races)
    train_end = int(n_races * train_ratio)
    train_races = unique_races[:train_end]
    val_races = unique_races[train_end:]
    train_df = df[df[group_col].isin(train_races)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_races)].reset_index(drop=True)
    test_df = val_df.copy()
    return train_df, val_df, test_df

# 年で時系列分割: year 未満をtrain, yearをval, year+1をtest
def time_series_group_split_by_year(df, year=2024, group_col="レースID"):
    years = df[group_col].astype(str).str[:4]
    df_train = df[years.astype(int) < year].reset_index(drop=True)
    df_val   = df[years.astype(int) == year].reset_index(drop=True)
    df_test  = df[years.isin([str(year+1)])].reset_index(drop=True)
    print("✅ データ分割完了")
    print(f"train年: <{year} ({df_train.shape[0]}件)")
    print(f"val年:   {year} ({df_val.shape[0]}件)")
    print(f"test年:  {year+1} ({df_test.shape[0]}件)")
    return df_train, df_val, df_test

# コース距離単位で勝率・連対率・複勝率・平均着順の統計特徴量を生成
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
    df = df.copy().sort_values(date_col).reset_index(drop=True)

    if add_course_distance and ("フィールド" in df.columns and "距離" in df.columns):
        df["コース距離"] = df["フィールド"].astype(str) + "_" + df["距離"].astype(str)

    df["is_win"] = (df[target_col] == 1).astype(int)
    df["is_ren"] = (df[target_col] <= 2).astype(int)
    df["is_fuku"] = (df[target_col] <= 3).astype(int)

    if combo_list is None:
        combo_list = []
        for r in range(1, max_comb + 1):
            combo_list += list(itertools.combinations(candidate_cols, r))
        if n_sample:
            combo_list = random.sample(combo_list, min(n_sample, len(combo_list)))

    winrate_mapping = {}

    # コース距離ごとに集計
    for course, df_course in tqdm(df.groupby("コース距離"), disable=not verbose, desc="コース距離単位処理"):
        df_course = df_course.copy()
        for cols in combo_list:
            name_base = "_".join(cols)
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
            prior_win = df_course["is_win"].mean()
            prior_ren = df_course["is_ren"].mean()
            prior_fuku = df_course["is_fuku"].mean()

            # 平滑化（ベイズ推定風）
            grouped[f"{name_base}_勝率"] = (grouped["win_sum"] + prior_win * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_連対率"] = (grouped["ren_sum"] + prior_ren * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_複勝率"] = (grouped["fuku_sum"] + prior_fuku * smooth_prior) / (grouped["total"] + smooth_prior)
            grouped[f"{name_base}_平均着順"] = grouped["avg_rank"]
            grouped[f"{name_base}_件数"] = grouped["total"]

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

    with open(save_mapping_path, "wb") as f:
        pickle.dump(winrate_mapping, f)

    print(f"✅ コース距離単位の統計特徴量生成完了。マッピングを {save_mapping_path} に保存しました。")
    return winrate_mapping, combo_list, feature_list

# 保存済みマッピングから統計特徴量を適用
def apply_stats_from_mapping(df, stats_dict, add_course_distance=True):
    df = df.copy()
    feature_list = []

    if add_course_distance and ("フィールド" in df.columns and "距離" in df.columns):
        if "コース距離" not in df.columns:
            df["コース距離"] = df["フィールド"].astype(str) + "_" + df["距離"].astype(str)

    all_features = sorted(list(stats_dict.keys()))
    print(f"適用対象特徴: {all_features}")

    def get_stat(course_dist, feature, value, stat_key):
        try:
            key_val = tuple(value) if isinstance(value, (list, tuple)) else value
            return stats_dict[str(feature)][(str(course_dist), key_val)][stat_key]
        except KeyError:
            return np.nan

    for feature in all_features:
        for stat_key, jp in zip(["win_rate", "place_rate", "show_rate", "n"], ["勝率", "連対率", "複勝率", "平均着順"]):
            col_name = f"{feature}_{jp}"
            feature_list.append(col_name)
            df[col_name] = df.apply(
                lambda r: get_stat(
                    r["コース距離"],
                    feature,
                    tuple(r[c] for c in feature.split('_')),
                    jp
                ),
                axis=1
            )

    return df, feature_list

# 分布近接度（KL/Jensen-Shannon/Wasserstein）で類似した過去開催を選択
def select_similar_train_races(
    df,
    group_col="レースID",
    feature_cols=None,
    target_year=2024,
    distance_type="kl",
    threshold=0.1,
    verbose=True,
):
    df = df.copy()
    df["year"] = df[group_col].astype(str).str[:4].astype(int)
    df["kaisaikai"] = df[group_col].astype(str).str[4:6].astype(int)
    df["year_meet"] = df["year"].astype(str) + "-" + df["kaisaikai"].astype(str).str.zfill(2)

    base_df = df[df["year"] == target_year]
    if base_df.empty:
        raise ValueError(f"{target_year}年のデータがありません。")

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in [group_col, "year", "kaisaikai", "year_meet"] and df[c].dtype != "object"
        ]

    dist_list = []
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
                    dist = wasserstein_distance(base_vals, grp_vals)
                else:
                    raise ValueError(f"distance_type={distance_type} は未対応")
            else:
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
