import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.0f}".format)


def load_csv(path):
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
    レース内の rank / relative / z-score を一括生成
    """
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
    df,
    group_distance_col="距離",
    group_field_col="フィールド",
    group_baba_col="馬場",
    n_past=5,
    score_col="スピード指数",
):
    """
    過去走条件と現在レース条件の一致度から適性スコアを生成
    """
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
    """
    勝ち馬（過去着順==1）だけの統計ベースラインを作成
    """
    past_rows = []
    for n in range(1, n_past + 1):
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
        tmp = tmp[tmp["過去着順_base"] == 1]
        past_rows.append(tmp)

    past_df = pd.concat(past_rows, ignore_index=True)

    baseline = (
        past_df
        .groupby(["場所_base", "距離_base", "フィールド_base", "クラス_base", "馬場_base"])
        [["後3F_base", "タイム_base", "スピード指数_base", "馬体重_base", "コーナー通過順_base", "馬番_base", '斤量_base']]
        .mean()
        .reset_index()
    )
    baseline.to_csv(f"./csv/{field}_winner_baseline.csv", index=False)
    return baseline


def add_past_diff_features(df, baseline, n_past=5):
    """
    各馬の過去走と baseline の差分 + 派生特徴量（平均・最大・最小・指数加重平均）
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


def add_fuku_payout_maxonly(df_pred, df_payout, race_col="レースID"):
    """
    複勝払戻のうち、各レース最大払戻の馬だけを正解とする（実験用）
    """
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


if __name__ == "__main__":
    field = 'chukyo'
    csv_path = f'./csv/df_all_{field}_2025.csv'

    df = load_csv(csv_path)
    df_pay = pd.read_csv(f'./csv/{field}_payouts_2025.csv')
    df_pay = df_pay.sort_values(['レースID'], ascending=[True])
    df = add_fuku_payout(df, df_pay)

    df['best後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).min(axis=1)
    df['av後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).mean(axis=1)

    df = add_race_condition_scores(df)

    base = build_winner_baseline(df, field)
    df, cols_diff, score_cols = add_past_diff_features(df, base)

    cols = cols_diff + score_cols
    df, cols_rank = add_race_relative_features(df, cols)

    feature_cols = (['フィールド適性スコア', "馬場適性スコア", "距離適性スコア",
                     '着順', '単勝オッズ', '距離', 'フィールド', '馬場', '馬単',
                     'レースID', '馬番', '人気', '複勝払戻', '複勝_hit',
                     '父馬', '騎手', '間隔', '性', '齢', '1クラス差', '1ペース差', 'オッズ']
                    + cols_rank)
    df.to_csv(f"./csv/df_all_{field}_2025_add.csv", index=True)
