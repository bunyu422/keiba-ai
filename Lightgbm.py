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


# file_path
###########################モデルごとに変更が必要############################
ninki = True
field = 'tokyo'
csv_path = f'./csv/df_all_tokyo_2025_add.csv'
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
        with open(f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl', "wb") as dd:
            pickle.dump(sire_mapping, dd)

        # val/test は train 全体の mapping を使う
        val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
        test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)
        df_2025['父馬_te'] = df_2025['父馬'].map(sire_mapping).fillna(-1)

        train_df, j_mapping = lw.target_encoding(train_df, '騎手', target_col)
        with open(f'./pickle-dict/jwin_dict_{field}_fold{fold}.pkl', "wb") as dd:
            pickle.dump(j_mapping, dd)

        # val/test は train 全体の mapping を使う
        val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
        test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)
        df_2025['騎手_te'] = df_2025['騎手'].map(j_mapping).fillna(-1)
        

        if ninki:
            feature_cols = [col for col in train_df.columns if col not in ["馬番", "is_fuku", "is_ren", 'コース距離', 'label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
        else:
            feature_cols = [col for col in train_df.columns if col not in ["is_fuku", "is_ren", 'コース距離', '人気', 'label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

        # train_df, temp = select_similar_train_races_at_k(pd.concat([train_df, val_df], axis=0, ignore_index=True), feature_cols=feature_cols, target_year=year, top_k=5)

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

        train_df = train_df.round(10)
        val_df = val_df.round(10)
        test_df = test_df.round(10)
        df_2025 = df_2025.round(10)

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

        # グルーピング
        train_df = train_df.sort_values(["レースID"]).reset_index(drop=True)
        train_list = train_df.groupby("レースID").size().to_list()
        val_df = val_df.sort_values(["レースID"]).reset_index(drop=True)
        eval_list = val_df.groupby("レースID").size().to_list()
        test_df = test_df.sort_values(["レースID"]).reset_index(drop=True)
        test_list = test_df.groupby("レースID").size().to_list()
        df_2025 = df_2025.sort_values(["レースID"]).reset_index(drop=True)
        df_2025_list = df_2025.groupby("レースID").size().to_list()

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
            # "objective": "binary", "metric": "binary_logloss",
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            'learning_rate': rate,
            'random_state': seed,
            'verbose_eval': 1000,
            # 'objective': 'lambdarank',
            # 'metric': 'ndcg',
            # 'ndcg_eval_at': [1],  # NDCG@1, @3, @5, @10 を同時に計算
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
            categorical_feature=cat_list,
            early_stopping_rounds=20,  # ← ここで指定
            num_boost_round=10000,      # ← イテレーション上限
            callbacks=[lgb.log_evaluation(period=0)]
        )

        tuner.run()
        # get_best_booster() は使えないので best_params を取得する
        best_params = tuner.best_params
        model = tuner.get_best_booster()
        # 最適パラメータをマージ
        # final_params = {**params, **best_params}

        # 学習
        # model = lgbm.train(
        #     final_params,
        #     lgb_train,  # トレーニングデータの指定
        #     valid_sets=[lgb_eval],
        #     # categorical_feature = cat_list,
        #     num_boost_round=10000,
        #     callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),lgb.log_evaluation(period=0)]
        # )

        # pklファイルとしてモデルを保存
        with open(f"./model/{field}_first_model_lgb_{model_type}_{fold}.pickle", "wb") as mk:
            pickle.dump(model, mk)

        # 学習後のモデル
        importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
        importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

        feature_names = train_df[feature_cols].columns

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
        y_pred = model.predict(val_df[feature_cols])
        val_df['pred_score'] = y_pred

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(test_df[feature_cols])
        test_df['pred_score'] = y_pred

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(df_2025[feature_cols])
        df_2025['pred_score'] = y_pred

        top = df_2025.loc[df_2025.groupby('レースID')['pred_score'].idxmax()]

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

        df_2025['ex'] = df_2025['pred_score'] * df_2025['オッズ']
        top = df_2025.loc[df_2025.groupby('レースID')['ex'].idxmax()]

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

        print(f"\n[ex評価結果2025 ブートストラップ評価]")
        print(f"レース数: {len(top)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")


        test_df['ex'] = test_df['pred_score'] * test_df['オッズ']
        top = test_df.loc[test_df.groupby('レースID')['ex'].idxmax()]

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

        print(f"\n[ex評価結果test ブートストラップ評価]")
        print(f"レース数: {len(top)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

        top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]

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

        print(f"\n[top評価結果test ブートストラップ評価]")
        print(f"レース数: {len(top)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

        test_df.to_csv(f'./csv/{field}_result_lgb_test_{fold}.csv', index=False)
        df_2025.to_csv(f'./csv/{field}_result_lgb_2025_{fold}.csv', index=False)
        continue

        # # まず test_df の index を連番にする
        # test_df = test_df.reset_index(drop=True)

        # # 1段目モデルで予測済みの test_df に対して
        # train_idx, val_idx, test_idx = split_test_for_next_stage(test_df, group_col="レースID")

        # # 例えば次段モデル用に分割
        # val_df = test_df.loc[train_idx].copy()
        # next_test = test_df.loc[test_idx].copy()
        # test_df = test_df.loc[val_idx].copy()

        # # 次段モデル用に分割
        # val_df = val_df.sort_values(["レースID"]).reset_index(drop=True)
        # eval_list = val_df.groupby("レースID").size().to_list()
        # test_df = test_df.sort_values(["レースID"]).reset_index(drop=True)
        # test_list = test_df.groupby("レースID").size().to_list()
        # df_2025 = df_2025.sort_values(["レースID"]).reset_index(drop=True)


        # # print("test len",len(test_df))
        # val_df = add_score_diff_features(val_df).round(10)
        # test_df = add_score_diff_features(test_df).round(10)
        # next_test = add_score_diff_features(next_test).round(10)
        # df_2025 = add_score_diff_features(df_2025).round(10)
        # # print("test len",len(test_df))

        # if ninki:
        #     feature_cols = [col for col in val_df.columns if col not in ['label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
        # else:
        #     feature_cols = [col for col in val_df.columns if col not in ['人気', 'label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
        # joblib.dump(feature_cols,f"./pickle-dict/{field}_lgb_cols_second.pkl")
        # feature_cols = ['score_diff_prev','score_diff_next','score_minus_mean','score_minus_mean_std','rank_in_race']
        # feature_cols = [
        #     # 差分・順位系
        #     'rank_in_race',
        #     'score_diff_prev',
        #     'score_diff_next',
        #     'score_diff_top1',
        #     'score_diff_top3_mean',

        #     # 統計・分布系
        #     'score_mean',
        #     'score_std',
        #     'score_range',
        #     'score_cv',
        #     'score_minus_mean',
        #     'score_minus_mean_std',

        #     # 正規化・確率化
        #     'score_relative',
        #     'score_softmax',
        #     'score_z',

        #     # 分布特性（レース単位）
        #     'score_entropy',
        #     'score_top_mean',
        #     'score_bottom_mean',
        #     'score_top_bottom_diff',
        #     'score_top_ratio',
        #     'score_rank_gap_ratio'
        # ]

        # lgb_train = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], group=eval_list)
        # lgb_eval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=lgb_train, group=test_list)

        # for seed in range(1, 6): 
        #     print(f"seed: {seed}")
        #     print(f"fold: {fold}")
        #     params = {
        #         'task': 'train',
        #         'boosting_type': 'gbdt',
        #         # 'objective': 'regression',  # ←ここでランキング学習と指定！
        #         # 'metric': 'rmse',   # for lambdarank
        #         'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        #         'learning_rate': rate,
        #         'random_state': seed,
        #         'verbose_eval': 1000,
        #         'objective': 'lambdarank',
        #         'metric': 'ndcg',
        #         'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
        #         'label_gain': [0,3,5,10],
        #         'bagging_seed': seed,
        #         'feature_fraction_seed': seed,
        #         'data_random_seed': seed,
        #         'deterministic': True,        # LightGBM 3.3.0 以降で利用可能
        #         'force_col_wise': True,       # 再現性を高める（内部順序を固定）
        #         'num_threads': 1,             # 厳密再現のためスレッド固定
        #     }
        #     ####################################################################################

            
        #     tuner = lgb.LightGBMTuner(
        #         params,
        #         optuna_seed=seed,
        #         train_set=lgb_train,
        #         valid_sets=[lgb_eval],
        #         # categorical_feature=cat_list,
        #         early_stopping_rounds=20,  # ← ここで指定
        #         num_boost_round=10000,      # ← イテレーション上限
        #         callbacks=[lgb.log_evaluation(period=0)]
        #     )

        #     tuner.run()
        #     # get_best_booster() は使えないので best_params を取得する
        #     best_params = tuner.best_params
        #     model = tuner.get_best_booster()
        #     # 最適パラメータをマージ
        #     # final_params = {**params, **best_params}

        #     # 学習
        #     # model = lgbm.train(
        #     #     final_params,
        #     #     lgb_train,  # トレーニングデータの指定
        #     #     valid_sets=[lgb_eval],
        #     #     # categorical_feature = cat_list,
        #     #     num_boost_round=10000,
        #     #     callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),lgb.log_evaluation(period=0)]
        #     # )

        #     # pklファイルとしてモデルを保存
        #     with open(f"./model/{field}_second_model_lgb_{model_type}_seed{seed}_{fold}.pickle", "wb") as mk:
        #         pickle.dump(model, mk)

        #     # 学習後のモデル
        #     importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
        #     importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

        #     feature_names = val_df[feature_cols].columns

        #     feat_imp_df = pd.DataFrame({
        #         "feature": feature_names,
        #         "importance_gain": importance_gain,
        #         "importance_split": importance_split
        #     }).sort_values(by="importance_gain", ascending=False)

        #     print(feat_imp_df)  # 上位20特徴量

        #     # plt.figure(figsize=(10,6))
        #     # sns.barplot(x="importance_gain", y="feature", data=feat_imp_df)
        #     # plt.title("Top 20 Feature Importance (LambdaRank)")
        #     # plt.show()
        #     y_pred = model.predict(next_test[feature_cols])
        #     next_test[f'result{seed}'] = y_pred
        #     # print("test len",len(next_test))

        #     # テストデータの予測 (予測クラスを返す)
        #     y_pred = model.predict(df_2025[feature_cols])
        #     df_2025[f'result{seed}'] = y_pred

        #     top = df_2025.loc[df_2025.groupby('レースID')[f'result{seed}'].idxmax()]

        #     # ブートストラップ
        #     n_boot = 10000  # ブートストラップ試行回数
        #     roi_list = []
        #     acc_list = []

        #     for _ in range(n_boot):
        #         # レース単位でリサンプリング（復元抽出）
        #         sampled = top.sample(frac=1.0, replace=True)
                
        #         total_bet = len(sampled) * 100
        #         total_return = sampled['単勝オッズ'].sum()  # 的中時のみ払戻あり
                
        #         hit_count = sampled['is_win'].sum()
        #         roi = total_return / total_bet
        #         acc = hit_count / len(sampled)
                
        #         roi_list.append(roi)
        #         acc_list.append(acc)

        #     roi_arr = np.array(roi_list)
        #     acc_arr = np.array(acc_list)

        #     # 点推定
        #     mean_roi = roi_arr.mean()
        #     mean_acc = acc_arr.mean()

        #     # 95%信頼区間
        #     roi_ci = np.percentile(roi_arr, [2.5, 97.5])
        #     acc_ci = np.percentile(acc_arr, [2.5, 97.5])

        #     print(f"\n[top評価結果2025 ブートストラップ評価]")
        #     print(f"レース数: {len(top)}")
        #     print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        #     print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")


        target_col = 'label'
        lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], group=train_list)
        lgb_eval = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], reference=lgb_train, group=eval_list)
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
            'ndcg_eval_at': [1,3,5],  # NDCG@1, @3, @5, @10 を同時に計算
            'label_gain': [0,2,3,8,10,15],
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

        y_pred = model.predict(val_df[feature_cols])
        val_df['pred_score_second'] = y_pred

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(test_df[feature_cols])
        test_df['pred_score_second'] = y_pred

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(df_2025[feature_cols])
        df_2025['pred_score_second'] = y_pred

        # スコア付与用
        rank_values = [1.0, 0.8, 0.6, 0.4, 0.2]

        def assign_rank_score(group):
            # pred_score_second降順で並べて順位をつける
            group = group.sort_values("pred_score_second", ascending=False).reset_index(drop=True)
            # 上位5頭にスコアを付与
            group["rank_score"] = 0.0
            n = min(len(rank_values), len(group))
            group.loc[:n-1, "rank_score"] = rank_values[:n]
            return group

        df_2025 = df_2025.groupby("レースID", group_keys=False).apply(assign_rank_score)
        df_2025['odds_diff'] = df_2025['オッズ'] - df_2025['pred_score']
        df_2025['expected_value'] = df_2025['rank_score'] * np.log1p(df_2025['odds_diff'].clip(lower=0.001))


        df_2025 = df_2025.sort_values(
            by=['レースID', 'expected_value'],  # 第二ソートキーを指定
            ascending=[True, False]     # pred_scoreは降順、馬番は昇順
        ).reset_index(drop=True)

        top = df_2025.loc[df_2025.groupby('レースID')['expected_value'].idxmax()]

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


        df_2025.to_csv(f'./csv/{field}_result_lgb_{model_type}_2025_{fold}.csv', index=False)
        continue


####### 評価 ########
        # テストデータの予測 (予測クラスを返す)
        # print("test len",len(next_test))
        # y_pred = model.predict(next_test[feature_cols])
        # next_test['pred_score'] = y_pred
        # # print("test len",len(next_test))

        # # テストデータの予測 (予測クラスを返す)
        # y_pred = model.predict(df_2025[feature_cols])
        # df_2025['pred_score'] = y_pred
        # next_test['pred_score_second'] = next_test[['result1', 'result2', 'result3', 'result4', 'result5']].mean(axis=1).round(10)
        # print("test len",len(next_test))

        # テストデータの予測 (予測クラスを返す)
        # df_2025['pred_score_second'] = df_2025[['result1', 'result2', 'result3', 'result4', 'result5']].mean(axis=1).round(10)

        next_test['expected_value'] = next_test['pred_score_second'] * next_test['オッズ']
        selected = next_test.loc[next_test.groupby('レースID')['expected_value'].idxmax()]
        print("select len",len(selected))
        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()

        hit_count = (selected['着順'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[評価結果]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
        
        df_2025['expected_value'] = df_2025['pred_score_second'] * df_2025['オッズ']
        selected = df_2025.loc[df_2025.groupby('レースID')['expected_value'].idxmax()]

        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()

        hit_count = (selected['着順'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[評価結果2025]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

        top = df_2025.loc[df_2025.groupby('レースID')['pred_score_second'].idxmax()]

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

        top = next_test.loc[next_test.groupby('レースID')['pred_score_second'].idxmax()]

        # 1. 各レースで予想順位を付ける（スコアが高いほど1位）
        # next_test['pred_rank'] = next_test.groupby('レースID')['pred_score'] \
        #                             .rank(ascending=False, method='first')

        # # 2. 各レースで上位3頭を抽出
        # top3 = next_test[next_test['pred_rank'] <= 3].copy()

        # # 3. 人気との乖離を計算
        # # 人気は1が最も人気、数値が大きいほど低人気
        # # → 値が大きいほど「予想より人気が低い」＝過小評価されている
        # top3['pop_diff'] = top3['人気'] - top3['pred_rank']

        # # 4. 各レースでpop_diffが最大の馬（市場が最も過小評価している馬）を抽出
        # top = top3.loc[top3.groupby('レースID')['pop_diff'].idxmax()].reset_index(drop=True)

        total_bet = len(top) * 100
        total_return = top['単勝オッズ'].sum()

        hit_count = (top['着順'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[top評価結果]")
        print(f"レース数: {len(top)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(top):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

        # val_df.to_csv(f'./csv/{field}_result_lgb_{model_type}_val_{fold}.csv', index=False)
        next_test.to_csv(f'./csv/{field}_result_lgb_{model_type}_test_{fold}.csv', index=False)
        df_2025.to_csv(f'./csv/{field}_result_lgb_{model_type}_2025_{fold}.csv', index=False)