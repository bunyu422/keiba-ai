import glob
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
import statsmodels.api as sm

# 行・列ともに省略せず全て表示する設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))  # 安定化のために最大値を引く
            return e_x / e_x.sum()
        return softmax

def pick_ev_max_per_race(df, p_col="pred_score", odds_col="オッズ", race_col="レースID", T=0.5):
    """レースごとにEV最大の馬を1頭だけ抽出（EV = p * odds）。"""
    df = df.copy()

    # T = 0.6
    softmax_T = make_softmax_with_temperature(T)

    df['softmax_score'] = df.groupby('レースID')['pred_score'].transform(softmax_T)
    df['expected_value'] = df['softmax_score'] * df['オッズ']

    # df["expected_value"] = df[p_col] * df[odds_col]
    # EV最大のindexを取る（同値があると複数返ることがあるので最後に重複排除）
    idx = df.groupby(race_col)["expected_value"].idxmax()
    sel = df.loc[idx].copy()
    # 念のため race 重複排除（理論上不要だが安全策）
    sel = sel.sort_values(["レースID", "expected_value"], ascending=[True, False])
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
    stake=100,
    use_bootstrap=False,
    n_bootstrap=5000,
    ci=0.95,
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

    hits = int(sel["is_win"].sum())
    return_yen = float((sel["is_win"] * sel["オッズ"] * stake).sum())
    invest = n_selected_races * stake
    roi = return_yen / invest
    p_hat = hits / n_selected_races if n_selected_races > 0 else 0

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
    p_col="pred_score",
    odds_col="オッズ",
    race_col="レースID",
    stake=100,
    # 探索レンジ（必要に応じて調整）
    ev_min_grid=np.round(np.arange(1.0, 2.01, 0.1), 2),
    prob_min_grid=[0, 0.05, 0.1, 0.12, 0.15, 0.2],
    odds_max_grid=[8, 10, 12, 15, np.inf],
    ev_max_grid=[3, 4, 5, np.inf],
    odds_min_grid=[1.0, 1.5, 2, 3, 4]  # 例：大穴のみを買いたいなら [4, 6, 8] なども検証可
):
    # まずレースごとにEV最大を1頭選ぶ
    base = pick_ev_max_per_race(df, p_col=p_col, odds_col=odds_col, race_col=race_col)

    results = []
    for ev_min, prob_min, odds_max, ev_max, odds_min in product(
        ev_min_grid, prob_min_grid, odds_max_grid, ev_max_grid, odds_min_grid
    ):
        sel = filter_by_thresholds(
            base,
            ev_min=ev_min,
            prob_min=prob_min,
            odds_min=odds_min,
            odds_max=odds_max,
            ev_max=ev_max
        )
        metrics = evaluate_selection(sel, stake=stake, use_bootstrap=True)
        results.append({
            "ev_min": ev_min,
            "prob_min": prob_min,
            "odds_min": odds_min,
            "odds_max": odds_max,
            "ev_max": ev_max,
            **metrics
        })

    res_df = pd.DataFrame(results).sort_values(["roi", "selected_races"], ascending=[False, False])
    return res_df

# --- nested fold eval ---
def nested_fold_eval_temperature(
    df,
    fold_col='fold',
    group_col='レースID',
    temperature_grid=[0.2, 0.3, 0.4, 0.5, 0.6],
    ev_min_grid=np.round(np.arange(1.0, 2.1, 0.1), 2),
    prob_min_grid=[0.05, 0.1, 0.12, 0.15, 0.2],
    odds_min_grid=[1.0, 1.5, 2.0, 3.0, 4.0],
    odds_max_grid=[8, 10, 12, 15, np.inf],
    ev_max_grid=[3, 4, 5, np.inf],
    stake=100,
    min_selected=100
):
    folds = sorted(df[fold_col].unique())
    all_fold_results = []
    chosen_params = []

    # 温度×foldの総ループ数
    total_loops = len(temperature_grid) * len(folds)

    loop_counter = 0
    for T in temperature_grid:
        for test_fold in folds:
            loop_counter += 1
            print(f"\n[{loop_counter}/{total_loops}] Temperature {T}, Fold {test_fold}")

            train_df = df[df[fold_col] != test_fold].reset_index(drop=True)
            test_df  = df[df[fold_col] == test_fold].reset_index(drop=True)

            # 温度 T で一度だけ softmax 計算
            base_train = pick_ev_max_per_race(train_df, 'pred_score', 'オッズ', group_col, T)

            best = None
            best_roi = -np.inf
            total_combinations = len(ev_min_grid) * len(prob_min_grid) \
                                 * len(odds_min_grid) * len(odds_max_grid) * len(ev_max_grid)

            # 閾値探索
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
                    best = {
                        'temperature': T,
                        'ev_min': ev_min,
                        'prob_min': prob_min,
                        'odds_min': odds_min,
                        'odds_max': odds_max,
                        'ev_max': ev_max,
                        'train_selected': metrics['selected_races'],
                        'train_hits': metrics['hits'],
                        'train_roi': roi,
                        'train_roi_ci_low': metrics['roi_ci_low'],
                        'train_roi_ci_high': metrics['roi_ci_high']
                    }

            if best is None:
                print(f"No valid params for fold {test_fold} (T={T})")
                continue

            # test fold で検証
            base_test = pick_ev_max_per_race(test_df, 'pred_score', 'オッズ', group_col, T)
            sel_test = filter_by_thresholds(base_test, best['ev_min'], best['prob_min'],
                                            best['odds_min'], best['odds_max'], best['ev_max'])
            test_metrics = evaluate_selection(sel_test, stake=stake)
            test_metrics.update({'fold': test_fold, **best})

            all_fold_results.append(test_metrics)
            chosen_params.append(best)

    res_df = pd.DataFrame(all_fold_results)
    params_df = pd.DataFrame(chosen_params)

    # 各foldごとのROI上位5件
    top5_per_fold = res_df.groupby('fold', group_keys=False).apply(
        lambda x: x.nlargest(5, 'roi_ci_low')  # 下限値でソート
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



def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()
    
    # with open('./model/platt.pkl', 'rb') as f:
    #     platt = pickle.load(f)

    # val_df['pred_score'] = platt.predict_proba(np.array(val_df['pred_score']).reshape(-1, 1))[:, 1]

    
    # df_sorted = val_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df_sorted[['レースID', 'pred_score']].head(30))
    
    # val_df['log_odds'] = np.log(val_df['オッズ'] + 1)
    for i in range(1, 21):
        # T = 0.1 * i  # 例：温度を0.5に設定（小さいほど尖る）
        T = 0.1
        softmax_T = make_softmax_with_temperature(T)

        val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
        val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
        # val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
        top_by_race = val_df.groupby('レースID').apply(
            lambda df: df.sort_values('expected_value', ascending=False)
        )

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
        
        selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]
        # selected = selected[selected['expected_value'] < 4]
        # selected = selected[selected['オッズ'] > i]
        # print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])
        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()
        hit_count = (selected['着順'] == 1).sum()
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

if __name__ == '__main__':
    # # テスト
    # --- CSVの読み込みとfold列付与 ---
    # csv_files = [
    #     './csv/tokyo_result_listnet_0.csv',
    #     './csv/tokyo_result_listnet_1.csv',
    #     './csv/tokyo_result_listnet_2.csv',
    #     './csv/tokyo_result_listnet_3.csv',
    #     './csv/tokyo_result_listnet_4.csv'
    # ]

    # csv_files = [
    #     './csv/tokyo_result_listnet_mse80_0.csv',
    #     './csv/tokyo_result_listnet_mse80_1.csv',
    #     './csv/tokyo_result_listnet_mse80_2.csv',
    #     './csv/tokyo_result_listnet_mse80_3.csv',
    #     './csv/tokyo_result_listnet_mse80_4.csv'
    # ]

    csv_files = [
        './csv/tokyo_result_listnet2_mse80_0.csv',
        './csv/tokyo_result_listnet2_mse80_1.csv',
        './csv/tokyo_result_listnet2_mse80_2.csv',
        './csv/tokyo_result_listnet2_mse80_3.csv',
        './csv/tokyo_result_listnet2_mse80_4.csv'
    ]

    # csv_files = [
    #     './csv/tokyo_result_listnet_mse0_0.csv',
    #     './csv/tokyo_result_listnet_mse0_1.csv',
    #     './csv/tokyo_result_listnet_mse0_2.csv',
    #     './csv/tokyo_result_listnet_mse0_3.csv',
    #     './csv/tokyo_result_listnet_mse0_4.csv'
    # ]

    dfs = []
    for i, path in enumerate(csv_files):
        df_fold = pd.read_csv(path)
        df_fold['fold'] = i
        dfs.append(df_fold)
        print(f"レース数: {len(df_fold['レースID'].unique())}")

    df = pd.concat(dfs, ignore_index=True)
    # df = pd.read_csv('./csv/tokyo_result_listnet_mse80_0.csv')  # または val データ専用ファイルを読み込む
    # # レース数出力
    # print(f"レース数: {len(df['レースID'].unique())}")
    
    # 対象となる CSV ファイルパターン
    # csv_files = glob.glob('./csv/tokyo_result_listnet_*.csv')

    # # 読み込んで結合
    # df_list = [pd.read_csv(file) for file in csv_files]
    # df = pd.concat(df_list, ignore_index=True)
    # print(f"レース数: {len(df['レースID'].unique())}")

    # val_df_result = evaluate_model_on_val_df(df, model_path='./model/tokyo_listnet_0.pth', fold=0)

    # ========== 使い方 ==========
    # df は検証・テスト用データフレーム
    # 必須列: ["レースID", "pred_score", "オッズ", "is_win"]
    # pred_score は温度スケーリング後の確率（0~1推奨）

    # 例）グリッド探索して上位5条件を表示
    # res = grid_search_ev_policy(df)
    # print(res.head(5))

    res_df, params_df, summary, top = nested_fold_eval_temperature(df)
    print(summary)
    print(res_df)      # foldごとの test 評価
    print(params_df)   # foldごとの選ばれたパラメータ
    print('###############################################')
    print(top)

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
