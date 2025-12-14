import numpy as np
import pandas as pd

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)

def load_csv(path):
    # 学習データを読み込む
    df = pd.read_csv(path, index_col=0)
    return df

def add_race_relative_features(
    df,
    cols,
    group_col="レースID",
    add_rank=True,
    add_relative=True,
    add_zscore=True,
):
    """
    レース内での rank / relative / z-score を一括生成する。

    Parameters
    ----------
    df : pd.DataFrame
        入力データ
    cols : list of str
        特徴量カラム名のリスト（数値カラム）
    group_col : str
        レースID の列名
    add_rank : bool
        レース内順位特徴量を追加するか
    add_relative : bool
        平均からの差分特徴量を追加するか
    add_zscore : bool
        z-score（相対差 / 標準偏差）を追加するか

    Returns
    -------
    df : pd.DataFrame
        特徴量追加後の DataFrame
    """

    df = df.copy()
    col_list = []
    rank_cols = {}

    # グループごとの平均・標準偏差を事前に計算
    grouped = df.groupby(group_col)

    # 平均
    means = grouped[cols].transform('mean')
    # 標準偏差（0除算を避ける）
    stds = grouped[cols].transform('std').replace(0, np.nan)

    for col in cols:
        # rank：数値が大きいほど上位（ascending=False）
        if add_rank:
            rank_cols[f"{col}_rank"] = grouped[col].rank(ascending=False, method="dense")
            col_list.append(f"{col}_rank")

        # relative：平均との差
        if add_relative:
            rank_cols[f"{col}_rel"] = df[col] - means[col]
            col_list.append(f"{col}_rel")

        # zscore：相対値 / 標準偏差
        if add_zscore:
            rank_cols[f"{col}_z"] = (df[col] - means[col]) / stds[col]
            col_list.append(f"{col}_z")

    df = pd.concat([df, pd.DataFrame(rank_cols, index=df.index)], axis=1)

    return df, col_list

def add_race_condition_scores(
    df,
    group_distance_col="距離",
    group_field_col="フィールド",
    group_baba_col="馬場",
    n_past=5,
    score_col="スピード指数",   # or 1後3F, 1タイム, 着差などに変更可能
):
    """
    現在レース条件（距離・馬場・フィールド）と
    過去走条件の一致度に基づく適性スコアを自動生成する。

    Parameters
    ----------
    df : pd.DataFrame
    group_distance_col, group_field_col, group_baba_col : str
        現在のレース条件を表すカラム
    n_past : int
        過去走の数（1〜n の仕様に合わせる）
    score_col : str
        過去走の能力を表すカラム（スピード指数など）

    Returns
    -------
    df : pd.DataFrame
    """

    df = df.copy()

    # === 距離適性スコア ===
    dist_scores = []
    for i in range(1, n_past+1):
        past_dist = df[f"{i}距離"]
        # 距離差（小さいほど良い）→ マイナスでペナルティ
        diff = (past_dist - df[group_distance_col]).abs()

        # 能力 × 距離ペナルティ
        s = df[f"{i}{score_col}"] / (1 + diff / 200)  # 200m 差で 50% 減衰
        dist_scores.append(s)

    # 距離適性（平均）
    df["距離適性スコア"] = np.vstack(dist_scores).mean(axis=0)

    # === 馬場適性スコア ===
    baba_scores = []
    for i in range(1, n_past+1):
        # 完全一致：1.0
        same = (df[f"{i}馬場"] == df[group_baba_col]).astype(float)
        # 該当走のスコア × 一致度
        s = df[f"{i}{score_col}"] * (0.5 + 0.5 * same)
        # → 不一致時は0.5倍、一致で1倍（調整可）
        baba_scores.append(s)

    df["馬場適性スコア"] = np.vstack(baba_scores).mean(axis=0)

    # === フィールド適性（芝・ダート等） ===
    field_scores = []
    for i in range(1, n_past+1):
        same = (df[f"{i}フィールド"] == df[group_field_col]).astype(float)
        s = df[f"{i}{score_col}"] * (0.5 + 0.5 * same)
        field_scores.append(s)

    df["フィールド適性スコア"] = np.vstack(field_scores).mean(axis=0)

    return df

def build_winner_baseline(df, n_past=5):
    """
    過去走1〜5走前を縦結合し、勝ち馬だけで統計（baseline）を作る
    """
    past_rows = []

    for n in range(1, n_past+1):
        cols = {
            '場所_base': f'{n}場所',
            '距離_base': f'{n}距離',
            'フィールド_base': f'{n}フィールド',
            'クラス_base': f'{n}クラス',
            '馬場_base': f'{n}馬場',
            '過去着順_base': f'{n}過去着順',
            '後3F_base': f'{n}後3F',
            'タイム_base': f'{n}タイム',
            'スピード指数_base': f'{n}スピード指数',
            '馬体重_base': f'{n}馬体重',
            '馬番_base': f'{n}馬番',
            'コーナー通過順_base': f'{n}コーナー通過順',
            '斤量_base': f'{n}斤量',
        }

        tmp = df[[v for v in cols.values()]].copy()
        tmp.rename(columns={v: k for k, v in cols.items()}, inplace=True)
        tmp["走前"] = n

        # ★ここが修正点★
        # 統計に使うのは勝ち馬（過去着順 == 1）だけ
        tmp = tmp[tmp["過去着順_base"] == 1]

        past_rows.append(tmp)

    past_df = pd.concat(past_rows, ignore_index=True)

    # ★勝ち馬だけで baseline を作成
    baseline = (
        past_df
        .groupby(["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"])
        [["後3F_base", "タイム_base", "スピード指数_base", "馬体重_base", "コーナー通過順_base", "馬番_base", '斤量_base']]
        .mean()
        .reset_index()
    )
    # print(baseline.columns.values)

    # CSV で保存
    baseline.to_csv(f"./csv/{field}_winner_baseline.csv", index=False)

    return baseline

# def add_past_diff_features(df, baseline, n_past=5):
#     """
#     各馬の1〜5走前と baseline の差分（context特徴）を追加する
#     """
#     df2 = df.copy()
#     cols = []
#     # print(baseline['後3F'].head(30))

#     for n in range(1, n_past+1):
#         key_cols = [f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"]
        
#         # baseline とマージ
#         merged = df2.merge(
#             baseline,
#             left_on=key_cols,
#             right_on=["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"],
#             how="left",
#             suffixes=("", "_base")
#         )

#         # print(merged.columns.values)

#         # 差分特徴量
#         for col in ["後3F", "タイム", "スピード指数", "馬体重", "コーナー通過順", "馬番", '斤量']:
#             df2[f"{n}{col}_diff"] = merged[f"{n}{col}"] - merged[f"{col}_base"]
#             cols.append(f"{n}{col}_diff")

#     return df2, cols

def add_past_diff_features(df, baseline, n_past=5):
    """
    各馬の1〜5走前と baseline の差分（context特徴） +
    走ごとの過去走スコア +
    派生特徴量（平均・最大・最小・指数加重平均）
    """
    df2 = df.copy()
    diff_cols = []
    score_cols = []

    for n in range(1, n_past + 1):
        key_cols = [f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"]

        merged = df2.merge(
            baseline,
            left_on=key_cols,
            right_on=["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"],
            how="left",
            suffixes=("", "_base")
        ).reset_index(drop=True)

        df2 = df2.reset_index(drop=True)

        feature_list = ["後3F", "タイム", "スピード指数", "馬体重", "コーナー通過順", "馬番", "斤量"]
        score_items = []

        for col in feature_list:
            diff_col = f"{n}{col}_diff"
            df2[diff_col] = merged[f"{n}{col}"] - merged[f"{col}_base"]
            diff_cols.append(diff_col)

            diff_val = df2[diff_col]

            # diff → score に変換
            if col in ["後3F", "タイム"]:
                score = -diff_val
            elif col in ["スピード指数", "斤量"]:
                score = diff_val
            elif col in ["馬体重", "コーナー通過順", "馬番"]:
                score = diff_val.abs()
            else:
                score = 0

            score_items.append(score)

        # 個別スコア → 走ごとの総合スコア
        score_col = f"{n}_past_score"
        df2[score_col] = sum(score_items)
        score_cols.append(score_col)

    # ===============================
    # ここから派生特徴量
    # ===============================

    # 過去走スコアがそろったら集約
    score_df = df2[score_cols]

    # 平均・最大・最小・合計
    df2["past_score_mean"] = score_df.mean(axis=1)
    df2["past_score_max"] = score_df.max(axis=1)
    df2["past_score_min"] = score_df.min(axis=1)
    df2["past_score_sum"] = score_df.sum(axis=1)

    # 指数加重平均（最近重視）
    # 1走前: 重み=1.0, 2走前=exp(-α), 3走前=exp(-2α)...
    alpha = 0.7
    weights = np.array([np.exp(-alpha * (i)) for i in range(n_past)])

    # 正規化
    weights = weights / weights.sum()

    df2["past_score_ewm"] = (score_df.values * weights).sum(axis=1)

    score_cols.extend(["past_score_mean", "past_score_max", "past_score_min", "past_score_sum", "past_score_ewm"])

    return df2, diff_cols, score_cols

def add_fuku_payout(df_pred, df_payout, race_col="レースID"):
    # --- 複勝払戻だけ取り出す ---
    fuku = df_payout.query("券種 == '複勝'")[
        [race_col, "馬番", "払い戻し金額"]
    ].copy()

    # df_pred と型を合わせる
    fuku["馬番"] = fuku["馬番"].astype(float).astype(str)
    df_pred["馬番"] = df_pred["馬番"].astype(str)

    # print(fuku['馬番'].head())
    # print(df_pred['馬番'].head())

    # --- マージ ---
    merged = df_pred.merge(
        fuku,
        on=[race_col, "馬番"],
        how="left"
    )

    # --- 当たり判定と払戻 ---
    merged["複勝_hit"] = merged["払い戻し金額"].notna().astype(int)
    merged["複勝払戻"] = merged["払い戻し金額"].fillna(0)

    return merged

def add_fuku_payout_maxonly(df_pred, df_payout, race_col="レースID"):
    # --- 複勝払戻だけ取り出す ---
    fuku = df_payout.query("券種 == '複勝'")[
        [race_col, "馬番", "払い戻し金額"]
    ].copy()

    # df_pred と型を合わせる
    fuku["馬番"] = fuku["馬番"].astype(float).astype(str)
    df_pred["馬番"] = df_pred["馬番"].astype(str)

    # --- マージ ---
    merged = df_pred.merge(
        fuku,
        on=[race_col, "馬番"],
        how="left"
    )

    # -------------------------------
    # ① 通常の複勝（3着以内）
    # -------------------------------
    merged["複勝_hit"] = merged["払い戻し金額"].notna().astype(int)
    merged["複勝払戻"] = merged["払い戻し金額"].fillna(0)

    # -------------------------------
    # ② 最大払戻の馬だけに複勝_hit_max / 複勝払戻_max
    # -------------------------------
    merged["複勝_hit_max"] = 0
    merged["複勝払戻_max"] = 0

    results = []

    for race_id, g in merged.groupby(race_col):
        max_pay = g["払い戻し金額"].max()

        # 払戻データがないレース
        if pd.isna(max_pay) or max_pay == 0:
            results.append(g)
            continue

        # 最大払戻の馬の候補
        candidates = g[g["払い戻し金額"] == max_pay]

        # df_pred の "人気" で判定（値が小さいほど人気が高い）
        best = candidates.sort_values("人気", na_position="last").iloc[0]

        g2 = g.copy()
        idx = best.name

        g2.loc[idx, "複勝_hit_max"] = 1
        g2.loc[idx, "複勝払戻_max"] = max_pay

        results.append(g2)

    merged = pd.concat(results).sort_index()

    return merged

###########################################################################

###################
'''
cols_diff+score_colsのrank / relative / z-scoreが有力
'''
###################

field = 'hanshin'
csv_path = f'./csv/df_all_{field}_2025.csv'

df = load_csv(csv_path)
# print(df.columns.values.tolist())
# df_pay = pd.read_csv(f'./csv/tokyo_payouts_2025.csv')

# df_pay = df_pay.sort_values(['レースID'], ascending=[True])

# df = add_fuku_payout(df, df_pay)
# df = add_fuku_payout_maxonly(df, df_pay)

# print(df_pay['レースID'].head())
# print(df_pay['レースID'].tail())

# print(df["複勝払戻_max"].head(30))
# print(df["複勝_hit_max"].value_counts())

# df = df[df['レースID'] == 202305021004]
# df = df.sort_values('馬番', ascending=False)

# key_cols = []
# for n in range(1, 5 + 1):
#     key_cols.extend([f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"])

# with open("log.txt", "a", encoding="utf-8") as f:
#     f.write(df[key_cols].to_string())

# cols = ['斤量', '馬番', '1後3F', '1着差', '1クラス', '1スピード指数',
#         'best着差', 'best後3F', 'av後3F', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度', "フィールド適性スコア", "馬場適性スコア", "距離適性スコア"]
cols = []

feature_cols = ['フィールド適性スコア', "馬場適性スコア", "距離適性スコア", '着順', '単勝オッズ', '距離', 'フィールド', '馬場', '馬単', 'レースID', '馬番', '人気',
                '父馬', '騎手', '間隔', '性', '齢', '1クラス差', '1ペース差', 'オッズ']
# past_cols = ['1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2コーナー通過順', '2後3F', '2馬体重', '2体重増減', '2着差', '2クラス', '2スピード指数', '2距離差', '2場所変化', '2フィールド変化', '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3コーナー通過順', '3後3F', '3馬体重', '3体重増減', '3着差', '3クラス', '3スピード指数', '3距離差', '3場所変化', '3フィールド変化', '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4コーナー通過順', '4後3F', '4馬体重', '4体重増減', '4着差', '4クラス', '4スピード指数', '4距離差', '4場所変化', '4フィールド変化', '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5コーナー通過順', '5後3F', '5馬体重', '5体重増減', '5着差', '5クラス', '5スピード指数', '5距離差', '5場所変化', '5フィールド変化']

df['best後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).min(axis=1)
df['av後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).mean(axis=1)

course_cols = ['場所','距離','フィールド','馬場']
target_cols = ['騎手','馬番','1距離','1場所','1フィールド']
feature_cols = list(dict.fromkeys(feature_cols + course_cols + target_cols))

# 馬場適性スコアを追加
df = add_race_condition_scores(df)

# 過去走を評価
base = build_winner_baseline(df)
df, cols_diff, score_cols = add_past_diff_features(df, base)

cols.extend(cols_diff+score_cols)
# feature_cols.extend(cols_diff)
# print(feature_cols)

# df = df[df['レースID'] == 202305021004]
# df = df.sort_values('馬番', ascending=False)

# key_cols = []
# for n in range(1, 5 + 1):
#     key_cols.extend([f"{n}場所", f"{n}距離", f"{n}フィールド", f"{n}クラス", f"{n}馬場"])

# with open("log.txt", "a", encoding="utf-8") as f:
#     f.write(df[feature_cols].to_string())

# レース内の rank / relative / z-score を追加
df, cols_rank = add_race_relative_features(df, cols)
feature_cols.extend(cols_rank)
# print(feature_cols)
# print(df.columns.values)
# feature_cols = feature_cols + past_cols
df = df[feature_cols]
print(df[feature_cols].columns.values.tolist())

df.to_csv(f'./csv/df_all_{field}_2025_add.csv', index=True)