import glob
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import statsmodels.api as sm
from betacal import BetaCalibration
from sklearn.isotonic import IsotonicRegression
import bandit

# 行・列ともに省略せず全て表示する設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))  # 安定化のために最大値を引く
            return e_x / e_x.sum()
        return softmax

def pick_ev_max_per_race(df, p_col="pred_score", odds_col="オッズ", race_col="レースID", T=0.7, bet_type="単勝"):
    """レースごとにEV最大の馬を1頭だけ抽出（EV = p * odds）。"""
    df = df.copy()

    # T = 0.6
    # softmax_T = make_softmax_with_temperature(T)

    # df['softmax_score'] = df.groupby('レースID')['pred_score'].transform(softmax_T)
    df['softmax_score'] = df['pred_score']
    df['expected_value'] = df['softmax_score'] * df['オッズ']
    
    if bet_type != "ワイド":
        # df["expected_value"] = df[p_col] * df[odds_col]
        # EV最大のindexを取る（同値があると複数返ることがあるので最後に重複排除）
        idx = df.groupby(race_col)["expected_value"].idxmax()
        sel = df.loc[idx].copy()
        
        # 念のため race 重複排除（理論上不要だが安全策）
        sel = sel.sort_values(["レースID", "expected_value"], ascending=[True, False])
        sel = sel.drop_duplicates(subset=[race_col], keep="first")
    else:
        # EV最大3のindexを取る（同値があると複数返ることがあるので最後に重複排除）
        idx = (
            df.groupby(race_col, group_keys=False)
            .apply(lambda g: g["expected_value"].nlargest(3).index)
        )
        # flatten して list に変換
        idx = idx.explode().dropna().astype(int).to_list()

        sel = df.loc[idx].copy()
        
        sel = sel.sort_values(["レースID", "expected_value"], ascending=[True, False])
    
    return sel

def pick_ev_max_per_race_alpha(df, alpha=1, beta=1, gamma=1, race_col="レースID", bet_type="単勝"):
    """レースごとにEV最大の馬を1頭だけ抽出（EV = p * odds）。"""
    df = df.copy()

    df['expected_value'] = df['softmax_score'] * df['オッズ']
    df['score_alpha'] = (alpha * df['softmax_score']) + (beta * df['expected_value']) + (1 / df['オッズ'] * gamma)
    
    if bet_type != "ワイド":
        # df["expected_value"] = df[p_col] * df[odds_col]
        # EV最大のindexを取る（同値があると複数返ることがあるので最後に重複排除）
        idx = df.groupby(race_col)["expected_value"].idxmax()
        sel = df.loc[idx].copy()
        
        # 念のため race 重複排除（理論上不要だが安全策）
        sel = sel.sort_values(["レースID", "expected_value"], ascending=[True, False])
        sel = sel.drop_duplicates(subset=[race_col], keep="first")
    else:
        # EV最大3のindexを取る（同値があると複数返ることがあるので最後に重複排除）
        idx = (
            df.groupby(race_col, group_keys=False)
            .apply(lambda g: g["expected_value"].nlargest(3).index)
        )
        # flatten して list に変換
        idx = idx.explode().dropna().astype(int).to_list()

        sel = df.loc[idx].copy()
        
        sel = sel.sort_values(["レースID", "expected_value"], ascending=[True, False])
    
    return sel

def softmax_pred(df, T):
    # T = 0.6
    softmax_T = make_softmax_with_temperature(T)

    df['softmax_score'] = df.groupby('レースID')['pred_score'].transform(softmax_T)
    df['expected_value'] = df['softmax_score'] * df['オッズ']

    return df

def pick_score_max_per_race(df, p_col="pred_score", odds_col="オッズ", race_col="レースID", T=0.7, bet_type="単勝"):
    """レースごとにEV最大の馬を1頭だけ抽出（EV = p * odds）。"""
    df = df.copy()

    # T = 0.6
    softmax_T = make_softmax_with_temperature(T)

    df['softmax_score'] = df.groupby('レースID')['pred_score'].transform(softmax_T)
    df['expected_value'] = df['softmax_score'] * df['オッズ']

    # df["expected_value"] = df[p_col] * df[odds_col]
    # EV最大のindexを取る（同値があると複数返ることがあるので最後に重複排除）
    idx = df.groupby(race_col)['pred_score'].idxmax()
    sel = df.loc[idx].copy()
    # 念のため race 重複排除（理論上不要だが安全策）
    sel = sel.sort_values(["レースID", 'pred_score'], ascending=[True, False])
    sel = sel.drop_duplicates(subset=[race_col], keep="first")
    return sel

def pick_ev_max_per_race_calibrated(
    df,
    beta_cal=None,  # BetaCalibration モデル（fit済み or None）
    use_beta=False,
    p_col="pred_score",
    odds_col="オッズ",
    race_col="レースID",
    T=1.0
):
    """
    レースごとにEV最大の馬を1頭だけ抽出
    - use_beta=Trueなら beta_cal.transform を使用
    - use_beta=Falseなら softmax(T) を使用
    """
    df = df.copy()

    if use_beta and beta_cal is not None:
        # beta calibration で確率化
        df["softmax_score"] = beta_cal.predict(df[[p_col]].values)
    else:
        # temperature scaling で softmax
        softmax_T = make_softmax_with_temperature(T)
        df["softmax_score"] = df.groupby(race_col)[p_col].transform(softmax_T)

    # EV = 確率 × オッズ
    df["expected_value"] = df["softmax_score"] * df[odds_col]

    # レースごとに EV 最大の馬を抽出
    idx = df.groupby(race_col)["expected_value"].idxmax()
    sel = df.loc[idx].copy()

    # 念のため race 重複排除（安全策）
    sel = sel.sort_values([race_col, "expected_value"], ascending=[True, False])
    sel = sel.drop_duplicates(subset=[race_col], keep="first")

    return sel

def filter_by_thresholds(
    sel,
    ev_min=1.0,
    prob_min=0.0,
    odds_min=1.0,
    odds_max=np.inf,
    ev_max=np.inf
):
    """閾値でフィルタリング。"""
    cond = (
        (sel["expected_value"] >= ev_min) &
        (sel["expected_value"] <= ev_max) &
        (sel["softmax_score"] >= prob_min) &
        (sel["オッズ"] >= odds_min) &
        (sel["オッズ"] <= odds_max)
    )
    return sel.loc[cond].copy()


import statsmodels.api as sm
import numpy as np
import pandas as pd

def evaluate_selection(
    sel,
    df_payout=None,
    stake=100,
    use_bootstrap=False,
    n_bootstrap=5000,
    ci=0.95,
    bet_type="単勝",
    random_state=None
):
    """購入結果を集計。ROIの95%信頼区間は Wilson またはブートストラップで計算。"""
    n_selected_races = sel["レースID"].nunique()
    if n_selected_races == 0:
        return {
            "selected_races": 0,
            "hits": 0,
            "hit_rate": 0.0,
            "return_yen": 0,
            "investment_yen": 0,
            "roi": 0.0,
            "roi_ci_low": None,
            "roi_ci_high": None,
            "ci_method": "none"
        }

    if bet_type == "単勝":
        hits = int(sel["is_win"].sum())
        return_yen = float((sel["is_win"] * sel["オッズ"] * stake).sum())
        invest = n_selected_races * stake
        roi = return_yen / invest
        p_hat = hits / n_selected_races if n_selected_races > 0 else 0

    elif bet_type == "複勝":
        # NaN を 0 に埋めて扱いやすくする
        sel["複勝払戻"] = sel["複勝払戻"].fillna(0)
        # 的中判定（払戻が入っていれば的中）
        sel["hit"] = (sel["複勝払戻"] > 0).astype(int)
        # 払戻はそのまま金額
        sel["payout"] = sel["複勝払戻"]
        hits = sel["hit"].sum()
        return_yen = float(sel["payout"].sum())
        invest = n_selected_races * stake
        roi = return_yen / invest
        p_hat = hits / n_selected_races if n_selected_races > 0 else 0
    elif bet_type == "ワイド":
        race_col = "レースID"

        # --- 各レースの上位3頭をリスト化 ---
        horse_lists = sel.groupby(race_col)["馬番"].apply(list)

        # --- 各レースで「本命流し」の2点だけ生成 ---
        bets = []
        for race_id, horses in horse_lists.items():
            if len(horses) >= 3:
                main = horses[0]      # 1着予想馬（sel内の先頭）
                others = horses[1:3]  # 2着・3着予想馬
                for o in others:
                    pair = "-".join(map(str, sorted([main, o])))
                    bets.append((race_id, pair))

        df_bets = pd.DataFrame(bets, columns=[race_col, "馬番"])

        if df_bets.empty:
            return 0, 0, 0, 0, 0, df_bets

        # --- 払戻データ（ワイドのみ抽出） ---
        payout_wide = df_payout.query("券種 == 'ワイド'")[
            [race_col, "馬番", "払い戻し金額"]
        ]

        # --- mergeで結合 ---
        df_merged = df_bets.merge(payout_wide, on=[race_col, "馬番"], how="left")

        # --- 的中判定 & 集計 ---
        df_merged["hit"] = df_merged["払い戻し金額"].notna().astype(int)
        df_merged["payout"] = df_merged["払い戻し金額"].fillna(0)

        hits = df_merged["hit"].sum()
        return_yen = df_merged["payout"].sum()
        invest = len(df_merged) * stake
        roi = return_yen / invest if invest > 0 else 0
        p_hat = hits / len(df_merged) if len(df_merged) > 0 else 0
        n_selected_races = df_bets[race_col].nunique()  # 実際に購入したレース数


    roi_ci_low, roi_ci_high, ci_method = None, None, "none"

    if hits > 0:
        if use_bootstrap:
            # --- 高速ブートストラップ法 ---
            rng = np.random.default_rng(random_state)

            # 各レースごとの払戻金（的中ならオッズ×stake、外れなら0）
            payouts = sel.groupby("レースID").apply(
                lambda g: float((g["is_win"] * g["オッズ"] * stake).sum())
            ).values

            n = len(payouts)
            # インデックスをまとめて生成
            idx = rng.integers(0, n, size=(n_bootstrap, n))
            sampled_payouts = payouts[idx]  # shape = (n_bootstrap, n)
            rois = sampled_payouts.sum(axis=1) / (n * stake)

            roi_ci_low = np.percentile(rois, (1-ci)/2 * 100)
            roi_ci_high = np.percentile(rois, (1+ci)/2 * 100)
            ci_method = "bootstrap"
        else:
            # --- Wilson法 ---
            ci_low, ci_high = sm.stats.proportion_confint(
                hits, n_selected_races, alpha=1-ci, method="wilson"
            )
            avg_payout = return_yen / hits
            roi_ci_low = ci_low * avg_payout / stake
            roi_ci_high = ci_high * avg_payout / stake
            ci_method = "wilson"
    else:
        roi_ci_low, roi_ci_high = 0.0, None
        ci_method = "no_hits"

    return {
        "selected_races": n_selected_races,
        "hits": hits,
        "hit_rate": p_hat,
        "return_yen": int(round(return_yen)),
        "investment_yen": invest,
        "roi": roi,
        "roi_ci_low": roi_ci_low,
        "roi_ci_high": roi_ci_high,
        "ci_method": ci_method
    }



def grid_search_ev_policy(
    df,
    df_payout=None,
    bet_type="単勝",
    fold_col='fold',
    p_col="pred_score",
    odds_col="オッズ",
    race_col="レースID",
    stake=100,
    temperature_grid=[0.2],
    ev_min_grid=np.round(np.arange(0, 2.01, 0.1), 2),
    prob_min_grid=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid=[8, 10, 12, 15, float('inf')],
    ev_max_grid=[3, 4, 5, float('inf')],
    odds_min_grid=[1.0, 1.5, 2, 3, 4],
    min_selected=0
):
    folds = sorted(df[fold_col].unique())
    all_results = []
    best_params_per_fold = []

    for fold in folds:
        fold_df = df[df[fold_col] == fold].reset_index(drop=True)
        best_roi = -float('inf')
        best_selected_races = 0
        best_params = None
        # 全パラメータの組み合わせ数を計算
        total = len(temperature_grid) * len(ev_min_grid) * len(prob_min_grid) \
                * len(odds_max_grid) * len(ev_max_grid) * len(odds_min_grid) \

        with tqdm(total=total, desc=f"Fold {fold}", unit="param") as pbar:
            for T in temperature_grid:
                # 温度 T で一度だけ softmax 適用
                base = pick_score_max_per_race(fold_df, p_col=p_col, odds_col=odds_col, race_col=race_col, T=T, bet_type=bet_type)
                # base = softmax_pred(fold_df, T)
                
                if bet_type == "ワイド":
                    # ---- ① 1着候補のみフィルタ ----
                    main_candidates = (
                        base.groupby(race_col, group_keys=False)
                        .head(1)  # 各レースの先頭行 = 1着予想馬
                    )

                for ev_min, prob_min, odds_max, ev_max, odds_min in product(
                    ev_min_grid, prob_min_grid, odds_max_grid, ev_max_grid, odds_min_grid
                ):
                    if bet_type == "ワイド":
                        main_filtered = filter_by_thresholds(
                            main_candidates,
                            ev_min=ev_min,
                            prob_min=prob_min,
                            odds_min=odds_min,
                            odds_max=odds_max,
                            ev_max=ev_max
                        )

                        # ---- ② 1着が残ったレースの上位3頭を購入対象にする ----
                        sel = base[base[race_col].isin(main_filtered[race_col])]
                    
                    else:
                        sel = filter_by_thresholds(
                            base,
                            ev_min=ev_min,
                            prob_min=prob_min,
                            odds_min=odds_min,
                            odds_max=odds_max,
                            ev_max=ev_max
                        )

                    # 選択レース数が少なければスキップ
                    if sel[race_col].nunique() < min_selected:
                        pbar.update(1)
                        continue

                    metrics = evaluate_selection(sel, df_payout=df_payout, stake=stake, use_bootstrap=False, bet_type=bet_type)
                    all_results.append({
                        "fold": fold,
                        "temperature": T,
                        "ev_min": ev_min,
                        "prob_min": prob_min,
                        "odds_min": odds_min,
                        "odds_max": odds_max,
                        "ev_max": ev_max,
                        **metrics
                    })

                    # ベストROI更新
                    # if metrics['roi'] is not None and metrics['roi'] > best_roi:
                    if metrics['roi'] > 1.0 and metrics['selected_races'] > best_selected_races:
                        best_roi = metrics['roi']
                        best_selected_races = metrics['selected_races']
                        best_params = {
                            "fold": fold,
                            "temperature": T,
                            "ev_min": ev_min,
                            "prob_min": prob_min,
                            "odds_min": odds_min,
                            "odds_max": odds_max,
                            "ev_max": ev_max,
                            "roi": metrics['roi'],
                            "roi_ci_low": metrics['roi_ci_low'],
                            "roi_ci_high": metrics['roi_ci_high'],
                            "selected_races": metrics['selected_races'],
                            "hits": metrics['hits'],
                            "hit_rate": metrics['hit_rate'],
                            "return_yen": metrics['return_yen'],
                            "investment_yen": metrics['investment_yen'],
                            "ci_method": metrics['ci_method']
                        }

                    pbar.update(1)

        if best_params is not None:
            best_params_per_fold.append(best_params)

    all_results_df = pd.DataFrame(all_results).sort_values(["fold", "roi_ci_low"], ascending=[True, False])
    best_params_df = pd.DataFrame(best_params_per_fold)

    return all_results_df, best_params_df

def grid_search_ev_policy_val_test(
    df_val,
    df_test,
    df_payout=None,
    bet_type="単勝",
    fold_col='fold',
    p_col="pred_score",
    odds_col="オッズ",
    race_col="レースID",
    stake=100,
    temperature_grid=[0.2],
    ev_min_grid=np.round(np.arange(0, 2.01, 0.1), 2),
    prob_min_grid=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid=[8, 10, 12, 15, float('inf')],
    ev_max_grid=[3, 4, 5, float('inf')],
    odds_min_grid=[1.0, 1.5, 2, 3, 4],
    min_selected=0
):
    """
    各foldで valデータで探索 → testデータで評価 する構成
    """

    folds = sorted(df_val[fold_col].unique())
    all_val_results = []
    all_test_results = []

    for fold in folds:
        val_fold_df = df_val[df_val[fold_col] == fold].reset_index(drop=True)
        test_fold_df = df_test[df_test[fold_col] == fold].reset_index(drop=True)

        # --- valでgrid search ---
        val_results, best_params_df = grid_search_ev_policy(
            val_fold_df,
            df_payout=df_payout,
            bet_type=bet_type,
            fold_col=fold_col,
            p_col=p_col,
            odds_col=odds_col,
            race_col=race_col,
            stake=stake,
            temperature_grid=temperature_grid,
            ev_min_grid=ev_min_grid,
            prob_min_grid=prob_min_grid,
            odds_max_grid=odds_max_grid,
            ev_max_grid=ev_max_grid,
            odds_min_grid=odds_min_grid,
            min_selected=min_selected
        )
        all_val_results.append(val_results)

        # --- best_paramsをtestに適用 ---
        best_params = best_params_df.iloc[0].to_dict()
        test_sel = pick_score_max_per_race(test_fold_df, T=best_params['temperature'],
                                        p_col=p_col, odds_col=odds_col,
                                        race_col=race_col, bet_type=bet_type)
        test_sel = filter_by_thresholds(
            test_sel,
            ev_min=best_params['ev_min'],
            prob_min=best_params['prob_min'],
            odds_min=best_params['odds_min'],
            odds_max=best_params['odds_max'],
            ev_max=best_params['ev_max']
        )

        test_metrics = evaluate_selection(test_sel, df_payout=df_payout,
                                          stake=stake, use_bootstrap=False,
                                          bet_type=bet_type)
        # test_metrics.update(best_params)
        print(best_params)
        test_metrics["fold"] = fold

        all_test_results.append(test_metrics)

    val_summary = pd.concat(all_val_results, ignore_index=True)
    test_summary = pd.DataFrame(all_test_results)

    return val_summary, test_summary


def grid_search_bandit(
    df,
    df_payout=None,
    bet_type="単勝",
    fold_col='fold',
    p_col="pred_score",
    odds_col="オッズ",
    race_col="レースID",
    stake=100,
    temperature_grid=[0.3],
    ev_min_grid=[1.1],
    prob_min_grid=[0],
    odds_max_grid=[float('inf')],
    ev_max_grid=[3],
    odds_min_grid=[3],
    temperature_grid2=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    ev_min_grid2=np.round(np.arange(0, 2.01, 0.1), 2),
    prob_min_grid2=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid2=[8, 10, 12, 15, float('inf')],
    ev_max_grid2=[3, 4, 5, float('inf')],
    odds_min_grid2=[1.0, 1.5, 2, 3, 4],
    # temperature_grid2=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    # ev_min_grid2=[0],
    # prob_min_grid2=[0],
    # odds_max_grid2=[float('inf')],
    # ev_max_grid2=[float('inf')],
    # odds_min_grid2=[1.0],
    min_selected=0 
):
    folds = sorted(df[fold_col].unique())
    all_results = []
    best_params_per_fold = []
    policies = [
    ("EV_max", bandit.policy_ev_max, {}),
    ("Prob_max", bandit.policy_prob_max, {}),
    ("Odds_range_3_8", bandit.policy_odds_range, {}),
    ("Top3_softmax", bandit.policy_top3_softmax_pick, {}),
    ("EV_Prob_combo", bandit.policy_ev_prob_combo, {}),
    ("Odds_range_1_4", bandit.policy_low_odds, {}),
    # 新規特徴量系
    ("Fastest_last3F", bandit.policy_fastest_last3f, {}),
    ("Highest_speed_index", bandit.policy_highest_speed_index, {}),
    ("Highest_uptrend", bandit.policy_highest_uptrend, {}),
    ]

    for fold in folds:
        # if fold != 3:
        #     continue
        fold_df = df[df[fold_col] == fold].reset_index(drop=True)
        best_roi = -float('inf')
        best_selected_races = 0
        best_params = None
        # 全パラメータの組み合わせ数を計算
        total = len(temperature_grid2) * len(ev_min_grid2) * len(prob_min_grid2) \
                * len(odds_max_grid2) * len(ev_max_grid2) * len(odds_min_grid2) \

        T = temperature_grid
        # 温度 T で一度だけ softmax 適用
        # base = pick_ev_max_per_race(fold_df, p_col=p_col, odds_col=odds_col, race_col=race_col, T=T, bet_type=bet_type)
        # base_df = softmax_pred(fold_df, T)
        # base = pick_ev_max_per_race_alpha(base_df)

        # if bet_type == "ワイド":
        #     # ---- ① 1着候補のみフィルタ ----
        #     main_candidates = (
        #         base.groupby(race_col, group_keys=False)
        #         .head(1)  # 各レースの先頭行 = 1着予想馬
        #     )

        # ev_min, prob_min, odds_max, ev_max, odds_min = ev_min_grid[0], prob_min_grid[0], odds_max_grid[0], ev_max_grid[0], odds_min_grid[0]
        # if bet_type == "ワイド":
        #     main_filtered = filter_by_thresholds(
        #         main_candidates,
        #         ev_min=ev_min,
        #         prob_min=prob_min,
        #         odds_min=odds_min,
        #         odds_max=odds_max,
        #         ev_max=ev_max
        #     )

        #     # ---- ② 1着が残ったレースの上位3頭を購入対象にする ----
        #     sel = base[base[race_col].isin(main_filtered[race_col])]
        
        # else:
        #     sel = filter_by_thresholds(
        #         base,
        #         ev_min=ev_min,
        #         prob_min=prob_min,
        #         odds_min=odds_min,
        #         odds_max=odds_max,
        #         ev_max=ev_max
        #     )

                    
        # main = sel.copy()
        with tqdm(total=total, desc=f"Fold {fold}", unit="param") as pbar:
            for T in temperature_grid2:
                base_df = softmax_pred(fold_df, T)
                # Bandit で選択
                # main_selected = main.copy()
                # dropped_races = set(base['レースID'].unique().tolist()) - set(main_selected['レースID'].unique().tolist())
                chosen = bandit.run_contextual_bandit_simulation(
                    # base_df[base_df['レースID'].isin(dropped_races)],  # ['レースID']dropped_races,
                    # base_df[base_df['レースID'].isin(dropped_races)],
                    base_df,
                    base_df,
                    policies

                )
                
                for ev_min, prob_min, odds_max, ev_max, odds_min in product(
                    ev_min_grid2, prob_min_grid2, odds_max_grid2, ev_max_grid2, odds_min_grid2
                ):
                    sel = filter_by_thresholds(
                        chosen,
                        ev_min=ev_min,
                        prob_min=prob_min,
                        odds_min=odds_min,
                        odds_max=odds_max,
                        ev_max=ev_max
                    )

                    # sel = pd.concat([main, con])

                    # 選択レース数が少なければスキップ
                    if sel[race_col].nunique() < min_selected:
                        pbar.update(1)
                        continue

                    metrics = evaluate_selection(sel, df_payout=df_payout, stake=stake, use_bootstrap=False, bet_type=bet_type)
                    all_results.append({
                        "fold": fold,
                        "temperature": T,
                        "ev_min": ev_min,
                        "prob_min": prob_min,
                        "odds_min": odds_min,
                        "odds_max": odds_max,
                        "ev_max": ev_max,
                        **metrics
                    })

                    # ベストROI更新
                    if metrics['roi_ci_low'] > 1.0 and metrics['selected_races'] > best_selected_races:
                        best_roi = metrics['roi_ci_low']
                        best_selected_races = metrics['selected_races']
                        best_params = {
                            "fold": fold,
                            "temperature": T,
                            "ev_min": ev_min,
                            "prob_min": prob_min,
                            "odds_min": odds_min,
                            "odds_max": odds_max,
                            "ev_max": ev_max,
                            "roi": metrics['roi'],
                            "roi_ci_low": metrics['roi_ci_low'],
                            "roi_ci_high": metrics['roi_ci_high'],
                            "selected_races": metrics['selected_races'],
                            "hits": metrics['hits'],
                            "hit_rate": metrics['hit_rate'],
                            "return_yen": metrics['return_yen'],
                            "investment_yen": metrics['investment_yen'],
                            "ci_method": metrics['ci_method']
                        }

                    pbar.update(1)

        if best_params is not None:
            best_params_per_fold.append(best_params)

    all_results_df = pd.DataFrame(all_results).sort_values(["fold", "roi_ci_low"], ascending=[True, False])
    best_params_df = pd.DataFrame(best_params_per_fold)

    return all_results_df, best_params_df




# --- nested fold eval ---
def nested_fold_eval_temperature(
    df,
    fold_col='fold',
    group_col='レースID',
    temperature_grid=[0.2, 0.3, 0.4, 0.5, 0.6],
    ev_min_grid=np.round(np.arange(0, 2.01, 0.1), 2),
    prob_min_grid=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid=[8, 10, 12, 15, float('inf')],
    ev_max_grid=[3, 4, 5, float('inf')],
    odds_min_grid=[1.0, 1.5, 2, 3, 4],
    stake=100,
    min_selected=100
):
    folds = sorted(df[fold_col].unique())
    all_fold_results = []
    chosen_params = []

    total_loops = len(folds) * len(temperature_grid)
    loop_counter = 0

    for test_fold in folds:
        loop_counter += 1
        print(f"\n[Outer Fold {test_fold} / {len(folds)}] Processing...")

        train_df = df[df[fold_col] != test_fold].reset_index(drop=True)
        test_df  = df[df[fold_col] == test_fold].reset_index(drop=True)

        best_inner = None
        best_roi = -np.inf

        # 温度ごとの Softmax 結果をキャッシュ
        softmax_cache = {}

        for T in temperature_grid:
            # 温度 T で一度だけ softmax 計算
            if T not in softmax_cache:
                softmax_cache[T] = pick_ev_max_per_race(train_df, 'pred_score', 'オッズ', group_col, T)
            base_train = softmax_cache[T]

            # inner CV で閾値探索
            for ev_min, prob_min, odds_min, odds_max, ev_max in product(
                ev_min_grid, prob_min_grid, odds_min_grid, odds_max_grid, ev_max_grid
            ):
                sel = filter_by_thresholds(base_train, ev_min, prob_min, odds_min, odds_max, ev_max)

                if sel[group_col].nunique() < min_selected:
                    continue

                metrics = evaluate_selection(sel, stake=stake)
                roi = metrics['roi_ci_low']

                if roi > best_roi:
                    best_roi = roi
                    best_inner = {
                        'temperature': T,
                        'ev_min': ev_min,
                        'prob_min': prob_min,
                        'odds_min': odds_min,
                        'odds_max': odds_max,
                        'ev_max': ev_max,
                        **metrics
                    }

        if best_inner is None:
            print(f"No valid params found for fold {test_fold}. Skipping...")
            continue

        # 最適化結果を test fold で評価
        base_test = pick_ev_max_per_race(test_df, 'pred_score', 'オッズ', group_col, best_inner['temperature'])
        sel_test = filter_by_thresholds(base_test, best_inner['ev_min'], best_inner['prob_min'],
                                        best_inner['odds_min'], best_inner['odds_max'], best_inner['ev_max'])
        test_metrics = evaluate_selection(sel_test, stake=stake)
        test_metrics.update({'fold': test_fold, **best_inner})

        all_fold_results.append(test_metrics)
        chosen_params.append(best_inner)

    res_df = pd.DataFrame(all_fold_results)
    params_df = pd.DataFrame(chosen_params)

    # 各foldごとのROI上位5件
    top5_per_fold = res_df.groupby('fold', group_keys=False).apply(
        lambda x: x.nlargest(5, 'roi_ci_low')
    )

    total_invest = res_df['investment_yen'].sum()
    total_return = res_df['return_yen'].sum()
    summary = {
        'folds': len(res_df),
        'mean_roi': res_df['roi'].mean() if len(res_df) else np.nan,
        'std_roi': res_df['roi'].std() if len(res_df) else np.nan,
        'total_invest': int(total_invest),
        'total_return': int(total_return),
        'overall_roi': total_return / total_invest if total_invest > 0 else np.nan
    }

    return res_df, params_df, summary, top5_per_fold


def nested_fold_eval_with_beta(
    df,
    df_val,  # ← validation データを追加
    fold_col='fold',
    group_col='レースID',
    use_beta=True,   # ← フラグで on/off 切り替え可能
    ev_min_grid=np.round(np.arange(0, 2.01, 0.1), 2),
    prob_min_grid=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid=[8, 10, 12, 15, float('inf')],
    ev_max_grid=[3, 4, 5, float('inf')],
    odds_min_grid=[1.0, 1.5, 2, 3, 4],
    stake=100,
    min_selected=100
):
    folds = sorted(df[fold_col].unique())
    all_fold_results = []
    chosen_params = []

    for test_fold in folds:
        print(f"\n[Outer Fold {test_fold} / {len(folds)}] Processing...")

        # outer 分割
        train_df = df[df[fold_col] != test_fold].reset_index(drop=True)
        test_df  = df[df[fold_col] == test_fold].reset_index(drop=True)
        val_df   = df_val[df_val[fold_col] == test_fold].reset_index(drop=True)  # ← foldに対応する val

        best_inner = None
        best_roi = -np.inf

        # ===== Beta calibration 学習 =====
        beta_cal = None
        if use_beta and len(val_df) > 0:
            beta_cal = BetaCalibration(parameters="abm")  # パラメータは要調整
            beta_cal.fit(val_df["pred_score"].values.reshape(-1,1), val_df["is_win"].values)

        # ===== 事前計算: Train側 確率化 + EV最大馬抽出 =====
        base_train = pick_ev_max_per_race_calibrated(train_df, beta_cal=beta_cal, use_beta=use_beta)

        # ===== Inner 探索 (閾値のみ) =====
        for ev_min, prob_min, odds_min, odds_max, ev_max in product(
            ev_min_grid, prob_min_grid, odds_min_grid, odds_max_grid, ev_max_grid
        ):
            sel = filter_by_thresholds(base_train, ev_min, prob_min, odds_min, odds_max, ev_max)
            if sel[group_col].nunique() < min_selected:
                continue

            metrics = evaluate_selection(sel, stake=stake)
            roi = metrics['roi_ci_low']

            if roi > best_roi:
                best_roi = roi
                best_inner = {
                    'ev_min': ev_min,
                    'prob_min': prob_min,
                    'odds_min': odds_min,
                    'odds_max': odds_max,
                    'ev_max': ev_max,
                    **metrics
                }

        if best_inner is None:
            print(f"No valid params found for fold {test_fold}. Skipping...")
            continue

        # ===== Test fold 評価 (一度だけ確率化 + EV抽出) =====
        base_test = pick_ev_max_per_race_calibrated(test_df, beta_cal=beta_cal, use_beta=use_beta)
        sel_test = filter_by_thresholds(base_test,
                                        best_inner['ev_min'], best_inner['prob_min'],
                                        best_inner['odds_min'], best_inner['odds_max'],
                                        best_inner['ev_max'])
        test_metrics = evaluate_selection(sel_test, stake=stake)
        test_metrics.update({'fold': test_fold, **best_inner})

        all_fold_results.append(test_metrics)
        chosen_params.append(best_inner)

    res_df = pd.DataFrame(all_fold_results)
    params_df = pd.DataFrame(chosen_params)

    return res_df, params_df


def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()
    
    # with open('./model/platt.pkl', 'rb') as f:
    #     platt = pickle.load(f)

    # val_df['pred_score'] = platt.predict_proba(np.array(val_df['pred_score']).reshape(-1, 1))[:, 1]

    
    # df_sorted = val_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df_sorted[['レースID', 'pred_score']].head(30))
    
    
    
    # val_df['log_odds'] = np.log(val_df['オッズ'] + 1)
    for i in range(1, 21):
        T = 1  # 例：温度を0.5に設定（小さいほど尖る）
        # T = i / 1000
        softmax_T = make_softmax_with_temperature(T)

        # val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
        # Xは出力スコア, yは1/0の勝ち負けラベル

        # X = val_df['pred_score']
        # y = val_df['is_win']

        # mask = ~y.isna()
        # X = X[mask].values.reshape(-1,1)
        # y = y[mask].values  # 1=勝ち, 0=負け

        # # base_estimator を先に fit する
        # base_clf = LogisticRegression()
        # base_clf.fit(X, y)  # これで "prefit" 状態になる

        # # Platt scaling (sigmoid)
        # platt = CalibratedClassifierCV(base_estimator=base_clf, method='sigmoid', cv='prefit')
        # platt.fit(X, y)

        # val_df['softmax_score'] = platt.predict_proba(X)[:, 1]

        # val_df['softmax_score'] = val_df['pred_score']
        # val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
        # # val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
        # top_by_race = val_df.groupby('レースID').apply(
        #     lambda df: df.sort_values('expected_value', ascending=False)
        # )

    #     print(top_by_race[['softmax_score', 'オッズ', 'expected_value', 'win_prob']].head(100))

        # a = 0.3
        # b = 0.7
        # val_df['expected_value'] = (val_df['softmax_score'] ** a) * (val_df['オッズ'] ** b)

        # val_df['expected_value'] = val_df['pred_score'] * val_df['log_odds']

        # 各レースごとに期待値上位3頭を取得
        # top3_ev = (
        #     val_df.sort_values(['レースID', 'pred_score'], ascending=[True, False])
        #         .groupby('レースID')
        #         .head(3)
        # )

        # # その中で pred_score 最大の馬を1頭だけ抽出
        # selected = (
        #     top3_ev.loc[top3_ev.groupby('レースID')['expected_value'].idxmax()]
        # )
        
        # print(df["pred"].head(30))
        selected = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
        # print(df["馬番"].head(10))
        
        # 検証用
        # selected['pred_score_scaled'] = (selected['pred_score'] - selected['pred_score'].mean()) / (selected['pred_score'].std() + 1e-9)
        # selected['pred_score_pos'] = np.exp(selected['pred_score_scaled'])
        # selected['ev_scaled'] = (selected['expected_value'] - selected['expected_value'].mean()) / (selected['expected_value'].std() + 1e-9)
        # selected['ev_pos'] = np.exp(selected['ev_scaled'])

        # print(f"mean: {selected['pred_score'].mean()}")
        # print(f"std: {selected['pred_score'].std()}")
        # threshold = selected['pred_score'].quantile(0.5)
        # ev_threshold = selected['expected_value'].quantile(0.1)
        
        # # 実データ用
        # global_mean = 5.422899237652069
        # global_std = 2.314727952052557

        # selected['pred_score_scaled'] = (selected['pred_score'] - global_mean) / (global_std + 1e-9)
        # selected['pred_score_pos'] = np.exp(selected['pred_score_scaled'])
        
        # # # # # 3. 下位除外
        # threshold = 0.9002883150000001
        selected = selected[selected['オッズ'] >= 2]
        # selected = selected[selected['pred_score'] > threshold]
        # selected = selected[selected['expected_value'] > ev_threshold]


        # print(f"threshold: {threshold}")
        # print(f"ev_threshold: {ev_threshold}")
        
        # selected = selected[selected['expected_value'] < 4]
        # selected = selected[selected['オッズ'] > i]
        # print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])
        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()
        hit_count = (selected['is_win'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[評価結果 - Fold {i}]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
        pass
    # print(selected[['softmax_score', 'log_odds', 'expected_value']].sort_values('expected_value', ascending=False).head(20))

    
    # top = top[top['expected_value'] > 100]

    # for i in range(1, 21):
    #     T = 0.2
    #     softmax_T = make_softmax_with_temperature(T)

    #     val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
    #     val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
    #     top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
    #     top = top[top['expected_value'] > 0.1 * i]
    #     total_bet = len(top) * 100
    #     total_return = top['単勝オッズ'].sum()
    #     hit_count = (top['着順'] == 1).sum()
    #     roi = total_return / total_bet

    #     print(f"\n[top評価結果{i}]")
    #     print(f"レース数: {len(top)}")
    #     print(f"的中数: {int(hit_count)}")
    #     print(f"的中率: {hit_count / len(top):.2%}")
    #     print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

    return val_df

def scale_pred_score_per_race(df):
    df = df.copy()
    df['pred_score_scaled'] = df.groupby('レースID')['pred_score'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-9)
    )
    df['pred_score_pos'] = np.exp(df['pred_score_scaled'])
    return df

if __name__ == '__main__':
    # # テスト
    # --- CSVの読み込みとfold列付与 ---
    # csv_files = [
    #     './csv/tokyo_result_listnet_mse50_0.csv',
    #     './csv/tokyo_result_listnet_mse50_1.csv',
    #     './csv/tokyo_result_listnet_mse50_2.csv',
    #     './csv/tokyo_result_listnet_mse50_3.csv',
    #     './csv/tokyo_result_listnet_mse50_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_ranknet_0.csv',
    #     './csv/tokyo_result_ranknet_1.csv',
    #     './csv/tokyo_result_ranknet_2.csv',
    #     './csv/tokyo_result_ranknet_3.csv',
    #     './csv/tokyo_result_ranknet_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_ranknet2_test_0.csv',
    #     './csv/tokyo_result_ranknet2_test_1.csv',
    #     './csv/tokyo_result_ranknet2_test_2.csv',
    #     './csv/tokyo_result_ranknet2_test_3.csv',
    #     './csv/tokyo_result_ranknet2_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_lambdarank_test_0.csv',
    #     './csv/tokyo_result_lambdarank_test_1.csv',
    #     './csv/tokyo_result_lambdarank_test_2.csv',
    #     './csv/tokyo_result_lambdarank_test_3.csv',
    #     './csv/tokyo_result_lambdarank_test_4.csv'
    # ]

    # dfs = []
    # for i, path in enumerate(csv_files):
    #     df_fold = pd.read_csv(path)
    #     df_fold['fold'] = i
    #     dfs.append(df_fold)
    #     print(f"レース数: {len(df_fold['レースID'].unique())}")

    # df = pd.concat(dfs, ignore_index=True)
    # print(df.head(20))

    # csv_files = [
    #     './csv/tokyo_result_ranknet2_val_0.csv',
    #     './csv/tokyo_result_ranknet2_val_1.csv',
    #     './csv/tokyo_result_ranknet2_val_2.csv',
    #     './csv/tokyo_result_ranknet2_val_3.csv',
    #     './csv/tokyo_result_ranknet2_val_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_ranknet_val_0.csv',
    #     './csv/tokyo_result_ranknet_val_1.csv',
    #     './csv/tokyo_result_ranknet_val_2.csv',
    #     './csv/tokyo_result_ranknet_val_3.csv',
    #     './csv/tokyo_result_ranknet_val_4.csv'
    # ]

    # csv_files = [
    #     './csv/hanshin_result_ranknet_test_0.csv',
    #     './csv/hanshin_result_ranknet_test_1.csv',
    #     './csv/hanshin_result_ranknet_test_2.csv',
    #     './csv/hanshin_result_ranknet_test_3.csv',
    #     './csv/hanshin_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/kyoto_result_ranknet_test_0.csv',
    #     './csv/kyoto_result_ranknet_test_1.csv',
    #     './csv/kyoto_result_ranknet_test_2.csv',
    #     './csv/kyoto_result_ranknet_test_3.csv',
    #     './csv/kyoto_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_ranknet_test_0.csv',
    #     './csv/tokyo_result_ranknet_test_1.csv',
    #     './csv/tokyo_result_ranknet_test_2.csv',
    #     './csv/tokyo_result_ranknet_test_3.csv',
    #     './csv/tokyo_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/ooi_result_ranknet_test_0.csv',
    #     './csv/ooi_result_ranknet_test_1.csv',
    #     './csv/ooi_result_ranknet_test_2.csv',
    #     './csv/ooi_result_ranknet_test_3.csv',
    #     './csv/ooi_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     "./csv/nakayama3_result_ranknet_test_0.csv",
    #     './csv/nakayama2_result_ranknet_test_1.csv',
    #     './csv/nakayama2_result_ranknet_test_2.csv',
    #     './csv/nakayama2_result_ranknet_test_3.csv',
    #     './csv/nakayama_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     "./csv/nakayama3_result_lgb_test_0.csv",
    #     # './csv/nakayama3_result_lgb_test_1.csv',
    #     # './csv/nakayama3_result_lgb_test_2.csv',
    #     # './csv/nakayama3_result_lgb_test_3.csv',
    #     # './csv/nakayama3_result_lgb_test_4.csv'
    # ]

    # csv_files = [
    #     f'./csv/nakayama3_result_stacking2_2025_0.csv',
    #     f'./csv/nakayama3_result_stacking2_2025_1.csv',
    #     f'./csv/nakayama3_result_stacking2_2025_2.csv',
    #     f'./csv/nakayama3_result_stacking2_2025_3.csv',
    #     f'./csv/nakayama3_result_stacking2_2025_4.csv',
    # ]

    
    # csv_files = [
    #     # f'./csv/nakayama3_result_ranknet_2025_0.csv',
    #     # f'./csv/nakayama3_result_ranknet_2025_1.csv',
    #     # "./csv/nakayama2_result_fold1.csv"
    # ]

    # csv_files = [
    #     "./csv/monbetu_result_fold0.csv",
        # './csv/monbetu_result_ranknet_test_1.csv',
        # './csv/monbetu_result_ranknet_test_2.csv',
        # './csv/monbetu_result_ranknet_test_3.csv',
        # './csv/monbetu_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    # #     # './csv/monbetu_result_ranknet_test_0.csv',
    # #     # './csv/monbetu_result_ranknet_test_1.csv',
    # #     # './csv/monbetu_result_ranknet_test_2.csv',
    #     # './csv/monbetu_result_ranknet_test_3.csv',
    # #     './csv/monbetu2_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/kasamatu_result_ranknet_test_0.csv',
    #     './csv/kasamatu_result_ranknet_test_1.csv',
    #     './csv/kasamatu_result_ranknet_test_2.csv',
    #     './csv/kasamatu_result_ranknet_test_3.csv',
    #     './csv/kasamatu_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/sonoda_result_ranknet_test_0.csv',
    #     './csv/sonoda_result_ranknet_test_1.csv',
    #     './csv/sonoda_result_ranknet_test_2.csv',
    #     './csv/sonoda_result_ranknet_test_3.csv',
    #     './csv/sonoda_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/saga_result_ranknet_test_0.csv',
    #     './csv/saga_result_ranknet_test_1.csv',
    #     './csv/saga_result_ranknet_test_2.csv',
    #     './csv/saga_result_ranknet_test_3.csv',
    #     './csv/saga_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/hunabasi_result_ranknet_test_0.csv',
    #     './csv/hunabasi_result_ranknet_test_1.csv',
    #     './csv/hunabasi_result_ranknet_test_2.csv',
    #     './csv/hunabasi_result_ranknet_test_3.csv',
    #     './csv/hunabasi_result_ranknet_test_4.csv'
    # ]

    # csv_files = [
    #     './csv/nagoya_result_ranknet_test_0.csv',
    #     './csv/nagoya_result_ranknet_test_1.csv',
    #     './csv/nagoya_result_ranknet_test_2.csv',
    #     './csv/nagoya_result_ranknet_test_3.csv',
    #     './csv/nagoya_result_ranknet_test_4.csv'
    # ]

    # dfs = []
    # for i, path in enumerate(csv_files):
    #     df_fold = pd.read_csv(path)
    #     df_fold['fold'] = i
    #     dfs.append(df_fold)
    #     print(f"レース数: {len(df_fold['レースID'].unique())}")

    # df = pd.concat(dfs, ignore_index=True)
    # print(df['is_win'])
    # print(df['pred_score'])
    # print(df['着順'])

    # csv_files = [
    #     f'./csv/nakayama3_result_stacking_test_0.csv'
    # ]

    # # csv_files = [
    # #     './csv/monbetu_result_ranknet_val_0.csv',
    # #     # './csv/monbetu_result_ranknet_test_1.csv',
    # #     # './csv/monbetu_result_ranknet_test_2.csv',
    # #     # './csv/monbetu_result_ranknet_test_3.csv',
    # #     # './csv/monbetu_result_ranknet_test_4.csv'
    # # ]

    # dfs = []
    # for i, path in enumerate(csv_files):
    #     df_fold = pd.read_csv(path)
    #     df_fold['fold'] = i
    #     dfs.append(df_fold)
    #     print(f"レース数: {len(df_fold['レースID'].unique())}")

    # df_val = pd.concat(dfs, ignore_index=True)

    # print(df['pred_score'].min())
    # print(df['pred_score'].max())
    # print(df_val['pred_score'].min())
    # print(df_val['pred_score'].max())

    # print(df['レースID'].min())
    # print(df['レースID'].max())
    # print(len(df['レースID'].unique()))
    # print(df_val['レースID'].min())
    # print(df_val['レースID'].max())
    # print(len(df_val['レースID'].unique()))

    # payout_df = pd.read_csv('./csv/nakayama_payouts_2012-2024.csv')

    # 複勝の払い戻しだけ抽出
    # fukusho = payout_df[payout_df["券種"] == "複勝"]

    # # 馬番をintに直す
    # fukusho["馬番"] = fukusho["馬番"].astype(int)

    # # 結合
    # df = df.merge(
    #     fukusho[["レースID", "馬番", "払い戻し金額"]],
    #     on=["レースID", "馬番"],
    #     how="left"
    # ).rename(columns={"払い戻し金額": "複勝払戻"})

    # print(df[["レースID", "馬番", "着順", "複勝払戻"]].head(20))

    # df_val = df_val.copy()
    # df_val['is_win'] = (df_val['着順'] == 1).astype(int)

    # res_df, params_df = nested_fold_eval_with_beta(df, df_val)
    # print(res_df)
    # print(params_df)   # foldごとの選ばれたパラメータ


    # df = pd.read_csv('./csv/nakayama_result_ranknet_test_0.csv')  # または val データ専用ファイルを読み込む
    # # レース数出力
    # print(f"レース数: {len(df['レースID'].unique())}")
    
    # 対象となる CSV ファイルパターン
    # csv_files = glob.glob('./csv/tokyo_result_listnet_*.csv')

    # # 読み込んで結合
    # df_list = [pd.read_csv(file) for file in csv_files]
    # df = pd.concat(df_list, ignore_index=True)
    # print(f"レース数: {len(df['レースID'].unique())}")

    # CSVファイル一覧
    csv_files = [
        './csv/nakayama3_result_stacking2_2025_0.csv',
        './csv/nakayama3_result_stacking2_2025_1.csv',
        './csv/nakayama3_result_stacking2_2025_2.csv',
        './csv/nakayama3_result_stacking2_2025_3.csv',
        './csv/nakayama3_result_stacking2_2025_4.csv',
    ]

    # 最初のCSVをベースに読み込み
    df = pd.read_csv(csv_files[0])[['レースID', '馬番', 'pred_score']].rename(columns={'pred_score': 'pred_score_0'})

    # 2つ目以降を順次マージ
    for i, file in enumerate(csv_files[1:], 1):
        tmp = pd.read_csv(file)[['レースID', '馬番', 'pred_score']].rename(columns={'pred_score': f'pred_score_{i}'})
        df = df.merge(tmp, on=['レースID', '馬番'], how='inner')  # inner推奨（全fold共通の馬のみ）

    # 各foldの予測スコアの平均を計算
    pred_cols = [c for c in df.columns if c.startswith('pred_score_')]
    df['pred_mean'] = df[pred_cols].mean(axis=1)

    val_df_result = evaluate_model_on_val_df(df, model_path='./model/tokyo_listnet_0.pth', fold=0)

    # ========== 使い方 ==========
    # df は検証・テスト用データフレーム
    # 必須列: ["レースID", "pred_score", "オッズ", "is_win"]
    # pred_score は温度スケーリング後の確率（0~1推奨）

    # df['pred_score'] = df.groupby('レースID')['pred_score'].transform(lambda x: x - x.min())
    # df_val['pred_score'] = df_val.groupby('レースID')['pred_score'].transform(lambda x: x - x.min())

    

    # df['pred_score'] = df.groupby('レースID')['pred_score'].transform(
    # lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    # )

    # 例）グリッド探索して上位5条件を表示
    # val_summary, test_summary = grid_search_ev_policy_val_test(df_val, df)
    

    # res, best = grid_search_ev_policy(df)
    # res, best = grid_search_bandit(df)
    # print(best)
    # print(val_summary)
    # print(test_summary)

    # res_df, params_df, summary, top = nested_fold_eval_temperature(df)
    # print(summary)
    # # print(res_df)      # foldごとの test 評価
    # print(params_df)   # foldごとの選ばれたパラメータ
    # print('###############################################')
    # print(top)

    # 例）ベスト条件での選択馬一覧も見たい場合
    # best = res.iloc[0]
    # base = pick_ev_max_per_race(df)
    # chosen = filter_by_thresholds(
    #     base,
    #     ev_min=best.ev_min,
    #     prob_min=best.prob_min,
    #     odds_min=best.odds_min,
    #     odds_max=best.odds_max,
    #     ev_max=best.ev_max
    # )
    # print(chosen.head(20))
    # print(evaluate_selection(chosen))
