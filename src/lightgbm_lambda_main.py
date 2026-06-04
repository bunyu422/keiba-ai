"""
LightGBM LambdaRank 学習スクリプト（listwise 同等データパイプライン使用）

使い方:
    python src/lightgbm_lambda_main.py

前提:
    - csv/df_all_{field}_2025_add.csv が存在する（build_dataset.py で生成）

出力:
    - csv/{field}_lgb_result_test_{fold}.csv
    - model/{field}_lambdarank_{fold}.txt
"""

import random
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.common import splits
from src.lightgbm import features
from sklearn.linear_model import LogisticRegression
from src.common.transform import make_smooth_relevance_labels, make_top2_labels

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
warnings.simplefilter('ignore')

# ------------------------------------------------------------------ #
# 設定（listwise_main.py と同一）
# ------------------------------------------------------------------ #

SEED = 4
N_FOLDS = 5
field = 'chukyo'
csv_path = f'./csv/df_all_{field}_2025_add.csv'

context_cat_cols = ['フィールド', '馬場']
context_num_cols = ['距離']

scale_cols = ['フィールド適性スコア', '馬場適性スコア', '距離適性スコア', '間隔', '1クラス差', '1ペース差', '父馬_te', '騎手_te',
              '距離グループ_父馬_te', '騎手_距離_te', '騎手_フィールド_te',
              '間隔クラス', '前走後3F_レース内順位', '馬体重_trend_slope',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z',
              '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z',
              'past_score_mean', 'past_score_max', 'past_score_min', 'past_score_sum', 'past_score_ewm',
              '同距離過去率', '同場所過去率',
              'past_score_mean_rank', 'past_score_mean_rel', 'past_score_mean_z',
              'past_score_max_rank', 'past_score_max_rel', 'past_score_max_z',
              'past_score_min_rank', 'past_score_min_rel', 'past_score_min_z',
              'past_score_sum_rank', 'past_score_sum_rel', 'past_score_sum_z',
              'past_score_ewm_rank', 'past_score_ewm_rel', 'past_score_ewm_z',
]

inversion_cols = []
common_cols = ['場所','距離','フィールド','馬場','騎手','馬番','1距離','1場所','1フィールド','距離グループ']

feature_category = ['父馬', '騎手', '性', '齢']

# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #

def main():
    features.set_seed(SEED)

    # --- 1. データ読み込み ---
    print(f"Loading: {csv_path}")
    df = features.load_csv(csv_path)

    # --- 新規特徴量（データリークなし） ---
    df = features.add_distance_group(df)
    df = features.add_interval_class(df)
    df = features.add_last3f_race_rank(df)
    df = features.add_weight_trend_slope(df)

    # scale_cols に存在しない列があれば除去
    global scale_cols
    scale_cols = [c for c in scale_cols if c in df.columns]

    # --- 1b. オッズ前処理（race-level features より先に） ---
    df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())

    # --- 1c. Race-level market features ---
    print("Computing race-level market features...")
    race_stats = df.groupby('レースID').agg(
        n_horses=('馬番', 'count'),
        fav_odds=('オッズ', 'min'),
        odds_mean=('オッズ', 'mean'),
        odds_std=('オッズ', 'std'),
        odds_min=('オッズ', 'min'),
        odds_max=('オッズ', 'max'),
    )
    race_stats['odds_range_log'] = np.log(race_stats['odds_max'] / (race_stats['odds_min'] + 1))
    race_stats['fav_win_prob'] = 1 / race_stats['fav_odds']
    race_stats['odds_cv'] = race_stats['odds_std'] / (race_stats['odds_mean'] + 1)
    df = df.merge(race_stats[['n_horses', 'odds_range_log', 'fav_win_prob', 'odds_cv']],
                  on='レースID', how='left')

    # --- 1d. Payouts CSV 読み込み（馬連/ワイド評価用） ---
    payouts_path = f'./csv/{field}_payouts_2025.csv'
    payouts = pd.read_csv(payouts_path, encoding='utf-8')
    payouts.columns = ['idx', 'レースID', '券種', '馬番', '払戻']
    print(f"  Loaded {len(payouts)} payout rows (全券種)")

    # --- 2. 前処理 ---
    df['着順'] = df['着順'].fillna(0)
    df['smooth_rel'] = make_smooth_relevance_labels(df)
    df['is_win'] = (df['着順'] == 1).astype(int)
    df['is_top3'] = (df['着順'].between(1, 3)).astype(int)
    df['is_top2'] = (df['着順'].between(1, 2)).astype(int)
    df = make_top2_labels(df)
    gain_list = [0, 1, 2]

    # --- 3. 時系列CV ---
    random.seed(1)
    np.random.seed(1)
    splits_list, df = splits.time_series_group_cv_3split_2025(df)

    fold_results = []
    fold_importance = []
    fold_importance_free = []
    fold_importance_clf = []

    # Benter ROI grids (per-fold {(α,β): roi} for cross-fold optimization)
    benter_roi_grids = {k: [] for k in ['win','win_free','top2','top2f']}

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits_list):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df   = df.loc[val_idx].reset_index(drop=True)
        test_df  = df.loc[test_idx].reset_index(drop=True)

        print(f"  train: {len(train_df)} rows, val: {len(val_df)} rows, test: {len(test_df)} rows")

        # --- 4a. Target Encoding ---
        train_df, sire_mapping = features.target_encoding(train_df, '父馬', '着順')
        val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
        test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)

        train_df, j_mapping = features.target_encoding(train_df, '騎手', '着順')
        val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
        test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)

        # 騎手×調教師コンビTE
        for _df in [train_df, val_df, test_df]:
            _df['combo_騎手_厩舎'] = _df['騎手'].astype(str) + '_' + _df['厩舎'].astype(str)
        train_df, combo_mapping = features.target_encoding(train_df, 'combo_騎手_厩舎', '着順')
        val_df['combo_騎手_厩舎_te'] = val_df['combo_騎手_厩舎'].map(combo_mapping).fillna(-1)
        test_df['combo_騎手_厩舎_te'] = test_df['combo_騎手_厩舎'].map(combo_mapping).fillna(-1)

        # 新規交互作用TE（fold内で計算）
        te_group_list = [
            ['距離グループ', '父馬'],
            ['騎手', '距離'],
            ['騎手', 'フィールド'],
        ]
        new_te_cols = []
        for group_cols in te_group_list:
            train_df, val_df, test_df, _, te_col, _ = features.add_interaction_te_fold(
                train_df, val_df, test_df, test_df, group_cols, target='着順'
            )
            new_te_cols.append(te_col)

        # --- 4b. 特徴量カラム定義 ---
        exclude_cols = set([
            'オッズ', '払い戻し金額', "複勝_hit_max", "複勝払戻_max", "複勝払戻", "複勝_hit",
            '人気', '馬番', '厩舎', '騎手_厩舎',
            'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel',
            'pred_rank', 'num_horses_bin', '単勝オッズ', '馬単',
            'score', 'win_flag', 'win_prob', 'is_win', 'is_top3', 'is_top2', 'win_prob_by_rank',
        ])

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        te_cols = ['父馬_te', '騎手_te', 'combo_騎手_厩舎_te'] + new_te_cols
        feature_cols.extend([c for c in te_cols if c not in feature_cols])

        # embedding_cols はカテゴリ特徴量として LightGBM に渡す（除外しない）
        cat_cols = [c for c in feature_category if c in feature_cols]
        feature_cols = list(dict.fromkeys(feature_cols))  # 重複除去・順序維持

        print(f"  features: {len(feature_cols)}, categorical: {cat_cols}")

        # odds除去版特徴量（市場由来列を除外）
        odds_free_exclude = {'odds_cv', 'odds_range_log', 'fav_win_prob', 'n_horses'}
        feature_cols_free = [c for c in feature_cols if c not in odds_free_exclude]
        print(f"  features_free: {len(feature_cols_free)} (odds-free)")

        # --- 4c. 欠損補完 ---
        train_df = features.fill_nan(train_df, feature_cols)
        val_df   = features.fill_nan(val_df, feature_cols)
        test_df  = features.fill_nan(test_df, feature_cols)

        # --- 5. LightGBM Dataset 準備 ---
        # group = レースごとの頭数（レースIDでソート済みであることが前提）
        train_group = train_df.groupby('レースID').size().tolist()
        val_group   = val_df.groupby('レースID').size().tolist()
        test_group  = test_df.groupby('レースID').size().tolist()

        # label: rank_label（整数、高いほど良い）
        lgb_train = lgb.Dataset(
            train_df[feature_cols],
            label=train_df['rank_label'],
            group=train_group,
            categorical_feature=cat_cols if cat_cols else 'auto',
            free_raw_data=False,
        )
        lgb_val = lgb.Dataset(
            val_df[feature_cols],
            label=val_df['rank_label'],
            group=val_group,
            reference=lgb_train,
            categorical_feature=cat_cols if cat_cols else 'auto',
            free_raw_data=False,
        )

        cat_cols_free = [c for c in cat_cols if c in feature_cols_free]
        lgb_train_free = lgb.Dataset(
            train_df[feature_cols_free],
            label=train_df['rank_label'],
            group=train_group,
            categorical_feature=cat_cols_free if cat_cols_free else 'auto',
            free_raw_data=False,
        )
        lgb_val_free = lgb.Dataset(
            val_df[feature_cols_free],
            label=val_df['rank_label'],
            group=val_group,
            reference=lgb_train_free,
            categorical_feature=cat_cols_free if cat_cols_free else 'auto',
            free_raw_data=False,
        )

        # --- 6. 学習 ---
        params = {
            'objective': 'lambdarank',
            'boosting_type': 'gbdt',
            'metric': 'ndcg',
            'ndcg_eval_at': [1, 3, 5],
            'label_gain': gain_list,
            'learning_rate': 0.01,
            'num_leaves': 63,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'random_state': SEED + fold,
        }

        print("  Training LambdaRank...")
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'valid'],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(100),
            ],
        )
        print(f"  Best iteration: {model.best_iteration}, best score: {model.best_score}")

        # モデル保存
        os.makedirs('./model', exist_ok=True)
        model.save_model(f'./model/{field}_lambdarank_{fold}.txt')

        # --- 7. 予測 ---
        test_df = test_df.copy()
        test_df['pred_score'] = model.predict(test_df[feature_cols])
        val_df = val_df.copy()
        val_df['pred_score'] = model.predict(val_df[feature_cols])

        # --- 7a. Odds-free LambdaRank ---
        print("  Training odds-free LambdaRank...")
        params_free = params.copy()
        params_free['random_state'] = SEED + fold + 1000
        model_free = lgb.train(
            params_free,
            lgb_train_free,
            valid_sets=[lgb_train_free, lgb_val_free],
            valid_names=['train', 'valid'],
            num_boost_round=2000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Odds-free best iteration: {model_free.best_iteration}")
        test_df['pred_score_free'] = model_free.predict(test_df[feature_cols_free])

        # --- 7b. Win classifier + Platt scaling ---
        print("  Training win classifier...")
        clf_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbose': -1,
            'random_state': SEED + fold + 100,
        }
        clf = lgb.train(
            clf_params,
            lgb.Dataset(train_df[feature_cols], label=train_df['is_win']),
            valid_sets=[lgb.Dataset(val_df[feature_cols], label=val_df['is_win'])],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Win classifier best iteration: {clf.best_iteration}")

        val_clf_raw = clf.predict(val_df[feature_cols])
        test_clf_raw = clf.predict(test_df[feature_cols])

        calibrator = LogisticRegression()
        calibrator.fit(val_clf_raw.reshape(-1, 1), val_df['is_win'])
        test_win_prob = calibrator.predict_proba(test_clf_raw.reshape(-1, 1))[:, 1]
        test_df['win_prob'] = test_win_prob

        # --- 7c. Top-3 classifier + Platt scaling（ワイド用） ---
        print("  Training top-3 classifier...")
        clf_top3 = lgb.train(
            clf_params,
            lgb.Dataset(train_df[feature_cols], label=train_df['is_top3']),
            valid_sets=[lgb.Dataset(val_df[feature_cols], label=val_df['is_top3'])],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Top-3 classifier best iteration: {clf_top3.best_iteration}")

        val_top3_raw = clf_top3.predict(val_df[feature_cols])
        test_top3_raw = clf_top3.predict(test_df[feature_cols])

        cal_top3 = LogisticRegression()
        cal_top3.fit(val_top3_raw.reshape(-1, 1), val_df['is_top3'])
        test_df['top3_prob'] = cal_top3.predict_proba(test_top3_raw.reshape(-1, 1))[:, 1]

        # --- 7c-2. Top-2 classifier + Platt scaling（ワイド用） ---
        print("  Training top-2 classifier...")
        clf_top2 = lgb.train(
            clf_params,
            lgb.Dataset(train_df[feature_cols], label=train_df['is_top2']),
            valid_sets=[lgb.Dataset(val_df[feature_cols], label=val_df['is_top2'])],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Top-2 classifier best iteration: {clf_top2.best_iteration}")

        val_top2_raw = clf_top2.predict(val_df[feature_cols])
        test_top2_raw = clf_top2.predict(test_df[feature_cols])

        cal_top2 = LogisticRegression()
        cal_top2.fit(val_top2_raw.reshape(-1, 1), val_df['is_top2'])
        test_df['top2_prob'] = cal_top2.predict_proba(test_top2_raw.reshape(-1, 1))[:, 1]

        # --- 7d. Odds-free win classifier + Platt scaling ---
        print("  Training odds-free win classifier...")
        clf_win_free = lgb.train(
            clf_params,
            lgb.Dataset(train_df[feature_cols_free], label=train_df['is_win']),
            valid_sets=[lgb.Dataset(val_df[feature_cols_free], label=val_df['is_win'])],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Odds-free win classifier best iteration: {clf_win_free.best_iteration}")
        val_win_free_raw = clf_win_free.predict(val_df[feature_cols_free])
        test_win_free_raw = clf_win_free.predict(test_df[feature_cols_free])
        cal_win_free = LogisticRegression()
        cal_win_free.fit(val_win_free_raw.reshape(-1, 1), val_df['is_win'])
        test_df['win_prob_free'] = cal_win_free.predict_proba(test_win_free_raw.reshape(-1, 1))[:, 1]

        # --- 7d-2. Odds-free Top-2 classifier + Platt scaling ---
        print("  Training odds-free top-2 classifier...")
        clf_top2_free = lgb.train(
            clf_params,
            lgb.Dataset(train_df[feature_cols_free], label=train_df['is_top2']),
            valid_sets=[lgb.Dataset(val_df[feature_cols_free], label=val_df['is_top2'])],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(20, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        print(f"  Odds-free top-2 classifier best iteration: {clf_top2_free.best_iteration}")

        val_top2_free_raw = clf_top2_free.predict(val_df[feature_cols_free])
        test_top2_free_raw = clf_top2_free.predict(test_df[feature_cols_free])
        cal_top2_free = LogisticRegression()
        cal_top2_free.fit(val_top2_free_raw.reshape(-1, 1), val_df['is_top2'])
        test_df['top2_prob_free'] = cal_top2_free.predict_proba(test_top2_free_raw.reshape(-1, 1))[:, 1]

        # --- 7e. 特徴量重要度を記録（全モデル学習後に実行） ---
        imp = model.feature_importance(importance_type='gain')
        fold_importance.append(pd.Series(imp, index=feature_cols))
        imp_free = model_free.feature_importance(importance_type='gain')
        fold_importance_free.append(pd.Series(imp_free, index=feature_cols_free))
        imp_clf = clf.feature_importance(importance_type='gain')
        fold_importance_clf.append(pd.Series(imp_clf, index=feature_cols))

        # --- 8. 評価（listwise と同一の bootstrap） ---
        print(f"\n  [評価 Fold {fold+1}]")

        # 戦略A: pred_score 順位で top-1
        top_by_score = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()].copy()
        _run_bootstrap(top_by_score, f"top-pred_score")

        # 戦略B: expected_value = pred_score × odds で top-1
        test_df['expected_value'] = test_df['pred_score'] * test_df['オッズ']
        top_by_ev = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()].copy()
        _run_bootstrap(top_by_ev, f"top-expected_value")

        # 戦略C: model推薦 ≠ 一番人気のケースを分析
        race_groups = test_df.groupby('レースID')
        model_disagree = []
        model_agree = []
        for rid, group in race_groups:
            fav = group.loc[group['人気'].idxmin()]
            model_pick = group.loc[group['pred_score'].idxmax()]
            if fav.name != model_pick.name:
                model_disagree.append(model_pick)
            else:
                model_agree.append(model_pick)

        if model_disagree:
            _run_bootstrap(pd.DataFrame(model_disagree), f"model≠人気1 (n={len(model_disagree)})")
        if model_agree:
            _run_bootstrap(pd.DataFrame(model_agree), f"model=人気1 (n={len(model_agree)})")

        # --- 9. Benter calibration (ranking用, ワイドROI最適化) ---
        val_market_prob = 1 / val_df['オッズ']
        val_clf_cal = calibrator.predict_proba(val_clf_raw.reshape(-1, 1))[:, 1]

        best_alpha, best_beta = 1.0, 1.0
        best_val_roi = -999
        roi_grid = {}
        for alpha in np.arange(0.5, 5.1, 0.5):
            for beta in np.arange(0.5, 5.1, 0.5):
                num = (val_clf_cal ** alpha) * (val_market_prob ** beta)
                den = num + ((1 - val_clf_cal) ** alpha) * ((1 - val_market_prob) ** beta)
                benter_prob = num / den
                val_temp = val_df.copy()
                val_temp['benter_prob'] = benter_prob
                val_roi = _compute_wide_top2_roi(val_temp, payouts, score_col='benter_prob')
                roi_grid[(alpha, beta)] = val_roi
                if val_roi > best_val_roi:
                    best_val_roi = val_roi
                    best_alpha, best_beta = alpha, beta

        print(f"  Best α={best_alpha:.1f}, β={best_beta:.1f}, val wide roi={best_val_roi:.2%}")
        benter_roi_grids['win'].append(roi_grid)

        test_market_prob = 1 / test_df['オッズ']
        num = (test_win_prob ** best_alpha) * (test_market_prob ** best_beta)
        den = num + ((1 - test_win_prob) ** best_alpha) * ((1 - test_market_prob) ** best_beta)
        test_df['benter_prob'] = num / den

        # Benter calibration for odds-free model
        val_win_free_cal = cal_win_free.predict_proba(val_win_free_raw.reshape(-1, 1))[:, 1]
        best_alpha_free, best_beta_free = 1.0, 1.0
        best_val_roi_free = -999
        roi_grid = {}
        for alpha in np.arange(0.5, 5.1, 0.5):
            for beta in np.arange(0.5, 5.1, 0.5):
                num = (val_win_free_cal ** alpha) * (val_market_prob ** beta)
                den = num + ((1 - val_win_free_cal) ** alpha) * ((1 - val_market_prob) ** beta)
                benter_prob = num / den
                val_temp = val_df.copy()
                val_temp['benter_prob'] = benter_prob
                val_roi = _compute_wide_top2_roi(val_temp, payouts, score_col='benter_prob')
                roi_grid[(alpha, beta)] = val_roi
                if val_roi > best_val_roi_free:
                    best_val_roi_free = val_roi
                    best_alpha_free, best_beta_free = alpha, beta
        print(f"  Odds-free Best α={best_alpha_free:.1f}, β={best_beta_free:.1f}, val wide roi={best_val_roi_free:.2%}")
        benter_roi_grids['win_free'].append(roi_grid)

        num_free = (test_df['win_prob_free'] ** best_alpha_free) * (test_market_prob ** best_beta_free)
        den_free = num_free + ((1 - test_df['win_prob_free']) ** best_alpha_free) * ((1 - test_market_prob) ** best_beta_free)
        test_df['benter_prob_free'] = num_free / den_free

        # Benter calibration for top2_prob (full features)
        val_top2_cal = cal_top2.predict_proba(val_top2_raw.reshape(-1, 1))[:, 1]
        best_alpha_t2, best_beta_t2 = 1.0, 1.0
        best_val_roi_t2 = -999
        roi_grid = {}
        for alpha in np.arange(0.5, 5.1, 0.5):
            for beta in np.arange(0.5, 5.1, 0.5):
                num = (val_top2_cal ** alpha) * (val_market_prob ** beta)
                den = num + ((1 - val_top2_cal) ** alpha) * ((1 - val_market_prob) ** beta)
                benter_prob = num / den
                val_temp = val_df.copy()
                val_temp['benter_prob'] = benter_prob
                val_roi = _compute_wide_top2_roi(val_temp, payouts, score_col='benter_prob')
                roi_grid[(alpha, beta)] = val_roi
                if val_roi > best_val_roi_t2:
                    best_val_roi_t2 = val_roi
                    best_alpha_t2, best_beta_t2 = alpha, beta
        print(f"  Top2 Best α={best_alpha_t2:.1f}, β={best_beta_t2:.1f}, val wide roi={best_val_roi_t2:.2%}")
        benter_roi_grids['top2'].append(roi_grid)

        num_t2 = (test_df['top2_prob'] ** best_alpha_t2) * (test_market_prob ** best_beta_t2)
        den_t2 = num_t2 + ((1 - test_df['top2_prob']) ** best_alpha_t2) * ((1 - test_market_prob) ** best_beta_t2)
        test_df['benter_prob_t2'] = num_t2 / den_t2

        # Benter calibration for top2_prob_free (odds-free)
        val_top2_free_cal = cal_top2_free.predict_proba(val_top2_free_raw.reshape(-1, 1))[:, 1]
        best_alpha_t2f, best_beta_t2f = 1.0, 1.0
        best_val_roi_t2f = -999
        roi_grid = {}
        for alpha in np.arange(0.5, 5.1, 0.5):
            for beta in np.arange(0.5, 5.1, 0.5):
                num = (val_top2_free_cal ** alpha) * (val_market_prob ** beta)
                den = num + ((1 - val_top2_free_cal) ** alpha) * ((1 - val_market_prob) ** beta)
                benter_prob = num / den
                val_temp = val_df.copy()
                val_temp['benter_prob'] = benter_prob
                val_roi = _compute_wide_top2_roi(val_temp, payouts, score_col='benter_prob')
                roi_grid[(alpha, beta)] = val_roi
                if val_roi > best_val_roi_t2f:
                    best_val_roi_t2f = val_roi
                    best_alpha_t2f, best_beta_t2f = alpha, beta
        print(f"  Odds-free Top2 Best α={best_alpha_t2f:.1f}, β={best_beta_t2f:.1f}, val wide roi={best_val_roi_t2f:.2%}")
        benter_roi_grids['top2f'].append(roi_grid)

        num_t2f = (test_df['top2_prob_free'] ** best_alpha_t2f) * (test_market_prob ** best_beta_t2f)
        den_t2f = num_t2f + ((1 - test_df['top2_prob_free']) ** best_alpha_t2f) * ((1 - test_market_prob) ** best_beta_t2f)
        test_df['benter_prob_t2f'] = num_t2f / den_t2f

        # --- 10. Wide戦略評価 ---
        print(f"\n  [Wide戦略 Fold {fold+1}]")

        # Baseline (pred_score)
        _eval_exotic(test_df, payouts, "wide_pred_all")
        # Baseline (benter_prob)
        _eval_exotic(test_df, payouts, "wide_benter_all", score_col='benter_prob')
        # Baseline (top3_prob)
        _eval_exotic(test_df, payouts, "wide_top3_all", score_col='top3_prob')
        # Odds-free
        _eval_exotic(test_df, payouts, "wide_pred_free", score_col='pred_score_free')
        _eval_exotic(test_df, payouts, "wide_benter_free", score_col='benter_prob_free')

        # Step 1: benter_prob blend (full + odds-free average)
        test_df['benter_prob_blend'] = (test_df['benter_prob'] + test_df['benter_prob_free']) / 2
        _eval_exotic(test_df, payouts, "wide_benter_blend", score_col='benter_prob_blend')

        # Step 2: Hybrid signal (pred_score z-score × benter_prob_free)
        test_df['pred_z'] = test_df.groupby('レースID')['pred_score'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        test_df['hybrid_z'] = test_df['pred_z'] * test_df['benter_prob_free']
        _eval_exotic(test_df, payouts, "wide_hybrid_z", score_col='hybrid_z')

        # Top-2 classifiers
        _eval_exotic(test_df, payouts, "wide_benter_t2", score_col='benter_prob_t2')
        _eval_exotic(test_df, payouts, "wide_benter_t2f", score_col='benter_prob_t2f')

        # 結果保存
        test_df.to_csv(f'./csv/{field}_lgb_result_test_{fold}.csv', index=False)

        fold_results.append(test_df)

    # --- 9. Benter 較正: val ROI grid を cross-fold 平均 → 最適α,βで再評価 ---
    print(f"\n{'='*60}")
    print("Benter Calibration: Cross-fold Val ROI Optimization")
    print(f"{'='*60}")

    benter_labels = {
        'win': 'benter_prob (win classifier)',
        'win_free': 'benter_prob_free (odds-free win)',
        'top2': 'benter_prob_t2 (top2 classifier)',
        'top2f': 'benter_prob_t2f (odds-free top2)',
    }
    prob_col_map = {
        'win': 'win_prob',
        'win_free': 'win_prob_free',
        'top2': 'top2_prob',
        'top2f': 'top2_prob_free',
    }

    crossfold_params = {}
    for key in ['win','win_free','top2','top2f']:
        grids = benter_roi_grids[key]
        # 各foldの最適ペア
        per_fold_best = []
        for g in grids:
            best = max(g, key=g.get)
            per_fold_best.append(best)
        per_fold_str = ', '.join([f"({a:.1f},{b:.1f})" for a, b in per_fold_best])
        # cross-fold平均ROI最大化
        all_keys = set()
        for g in grids:
            all_keys.update(g.keys())
        avg_roi = {}
        for k in all_keys:
            vals = [g.get(k) for g in grids if k in g]
            avg_roi[k] = np.mean(vals)
        best_pair = max(avg_roi, key=avg_roi.get)
        crossfold_params[key] = best_pair
        print(f"  {benter_labels[key]:45s}")
        print(f"    per-fold: {per_fold_str}")
        print(f"    cross-fold best: α={best_pair[0]:.1f}, β={best_pair[1]:.1f} (avg val roi={avg_roi[best_pair]:.2%})")

    # 再評価: cross-fold best α,β を各foldのtestデータに適用
    print(f"\n{'='*60}")
    print("Re-evaluation with cross-fold optimized α,β")
    print(f"{'='*60}")
    cv_results = {k: [] for k in ['win','win_free','top2','top2f']}
    for fold in range(N_FOLDS):
        test_path = f'./csv/{field}_lgb_result_test_{fold}.csv'
        if not os.path.exists(test_path):
            print(f"  WARNING: {test_path} not found, skipping re-eval")
            continue
        test_df = pd.read_csv(test_path)
        test_market_prob = 1 / test_df['オッズ']

        for key in ['win','win_free','top2','top2f']:
            cf_a, cf_b = crossfold_params[key]
            prob_col = prob_col_map[key]
            num = (test_df[prob_col] ** cf_a) * (test_market_prob ** cf_b)
            den = num + ((1 - test_df[prob_col]) ** cf_a) * ((1 - test_market_prob) ** cf_b)
            test_df['_benter_cv'] = num / den
            roi = _compute_wide_top2_roi(test_df, payouts, score_col='_benter_cv')
            cv_results[key].append(roi)

    if any(len(v) == N_FOLDS for v in cv_results.values()):
        print(f"\n  {'Signal':30s} {'Fold1':>7s} {'Fold2':>7s} {'Fold3':>7s} {'Fold4':>7s} {'Fold5':>7s} {'平均':>7s} {'レンジ':>7s}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
        for key in ['win','win_free','top2','top2f']:
            rois = cv_results[key]
            label = benter_labels[key]
            roi_strs = [f"{r:.1%}" for r in rois]
            avg = np.mean(rois)
            rng = max(rois) - min(rois)
            print(f"  {label:30s} {roi_strs[0]:>7s} {roi_strs[1]:>7s} {roi_strs[2]:>7s} {roi_strs[3]:>7s} {roi_strs[4]:>7s} {avg:.1%} {rng:.1%}")

    # --- 10. 全体集計 ---
    print(f"\n{'='*60}")
    print("All folds complete.")

    # 特徴量重要度 全fold平均
    if fold_importance:
        imp_df = pd.concat(fold_importance, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n{'='*60}")
        print("Feature Importance (LambdaRank, gain, 5-fold avg)")
        print(f"{'='*60}")
        for feat, score in imp_df.head(30).items():
            print(f"  {feat:40s} {score:>10.0f}")
        imp_df.to_csv(f'./csv/{field}_feature_importance.csv')

    if fold_importance_free:
        imp_free_df = pd.concat(fold_importance_free, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n{'='*60}")
        print("Feature Importance (odds-free LambdaRank, gain, 5-fold avg)")
        print(f"{'='*60}")
        for feat, score in imp_free_df.head(20).items():
            print(f"  {feat:40s} {score:>10.0f}")

    if fold_importance_clf:
        imp_clf_df = pd.concat(fold_importance_clf, axis=1).mean(axis=1).sort_values(ascending=False)
        print(f"\n{'='*60}")
        print("Feature Importance (win classifier, gain, 5-fold avg)")
        print(f"{'='*60}")
        for feat, score in imp_clf_df.head(20).items():
            print(f"  {feat:40s} {score:>10.0f}")
    print(f"{'='*60}")
    print(f"{'='*60}")


def _run_bootstrap(df_selected, label):
    """listwise と同じ bootstrap 評価を実行して表示"""
    n_boot = 10000
    roi_list, acc_list = [], []

    for _ in range(n_boot):
        sampled = df_selected.sample(frac=1.0, replace=True)
        total_bet = len(sampled) * 100
        total_return = sampled["単勝オッズ"].sum()
        hit_count = sampled["is_win"].sum()
        roi_list.append(total_return / total_bet)
        acc_list.append(hit_count / len(sampled))

    roi_arr = np.array(roi_list)
    acc_arr = np.array(acc_list)
    mean_roi = roi_arr.mean()
    mean_acc = acc_arr.mean()
    roi_ci = np.percentile(roi_arr, [2.5, 97.5])
    acc_ci = np.percentile(acc_arr, [2.5, 97.5])

    print(f"  [{label}]")
    print(f"    レース数: {len(df_selected)}")
    print(f"    的中率: {mean_acc:.2%} (95%CI: {acc_ci[0]:.2%} ~ {acc_ci[1]:.2%})")
    print(f"    回収率: {mean_roi:.2%} (95%CI: {roi_ci[0]:.2%} ~ {roi_ci[1]:.2%})")


def _eval_exotic(test_df, payouts, label, score_col='pred_score'):
    """馬連・ワイド評価"""
    # Build payout lookups
    umaren = payouts[payouts['券種'] == '馬連']
    wide   = payouts[payouts['券種'] == 'ワイド']

    def _parse_pair(s):
        a, b = s.split('-')
        return frozenset([int(a), int(b)])

    umaren_lookup = {}
    for _, r in umaren.iterrows():
        umaren_lookup[r['レースID']] = (_parse_pair(r['馬番']), r['払戻'])

    wide_lookup = {}
    for _, r in wide.iterrows():
        rid = r['レースID']
        pair = _parse_pair(r['馬番'])
        payout = r['払戻']
        wide_lookup.setdefault(rid, {})[pair] = payout

    umaren_hits = 0
    umaren_total = 0
    umaren_return = 0

    wide_top2_hits = 0
    wide_top2_return = 0
    wide_box_hits = 0
    wide_box_return = 0
    wide_box_races = 0

    for rid, group in test_df.groupby('レースID'):
        sorted_horses = group.sort_values(score_col, ascending=False)
        top2 = sorted_horses.head(2)
        top3 = sorted_horses.head(3)

        horse_nums_top2 = frozenset(top2['馬番'].astype(int))

        # 馬連: top-2 pair
        umaren_total += 1
        if rid in umaren_lookup:
            win_pair, payout = umaren_lookup[rid]
            if horse_nums_top2 == win_pair:
                umaren_hits += 1
                umaren_return += payout

        # ワイド top-2 (1 pair)
        if rid in wide_lookup:
            pairs = wide_lookup[rid]
            if horse_nums_top2 in pairs:
                wide_top2_hits += 1
                wide_top2_return += pairs[horse_nums_top2]

        # ワイド top-3 box (3 pairs)
        wide_box_races += 1
        horse_list = top3['馬番'].astype(int).tolist()
        model_pairs = [frozenset([horse_list[i], horse_list[j]])
                       for i in range(len(horse_list))
                       for j in range(i+1, len(horse_list))]
        if rid in wide_lookup:
            pairs = wide_lookup[rid]
            for mp in model_pairs:
                if mp in pairs:
                    wide_box_hits += 1
                    wide_box_return += pairs[mp]

    n = umaren_total
    if n > 0:
        roi = umaren_return / (n * 100)
        acc = umaren_hits / n
        print(f"  [{label}] 馬連 top2 n={n} hit={umaren_hits} acc={acc:.2%} roi={roi:.2%}")

    if n > 0:
        roi = wide_top2_return / (n * 100)
        acc = wide_top2_hits / n
        print(f"  [{label}] ワイド top2 n={n} hit={wide_top2_hits} acc={acc:.2%} roi={roi:.2%}")

    n = wide_box_races
    if n > 0:
        cost = n * 300
        roi = wide_box_return / cost
        print(f"  [{label}] ワイド top3box n={n} hit={wide_box_hits} pairs roi={roi:.2%}")


def _compute_wide_top2_roi(test_df, payouts, score_col='pred_score'):
    """ワイド top2 ROI を計算（print無し、戻り値float）。Benter grid search用"""
    wide = payouts[payouts['券種'] == 'ワイド']
    def _parse_pair(s):
        a, b = s.split('-')
        return frozenset([int(a), int(b)])
    wide_lookup = {}
    for _, r in wide.iterrows():
        wide_lookup.setdefault(r['レースID'], {})[_parse_pair(r['馬番'])] = r['払戻']
    total_return = 0
    n_races = 0
    for rid, group in test_df.groupby('レースID'):
        n_races += 1
        top2 = group.sort_values(score_col, ascending=False).head(2)
        pair = frozenset(top2['馬番'].astype(int))
        if rid in wide_lookup and pair in wide_lookup[rid]:
            total_return += wide_lookup[rid][pair]
    return total_return / (n_races * 100) if n_races > 0 else 0.0


if __name__ == '__main__':
    main()
