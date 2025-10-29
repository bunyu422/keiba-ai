import copy
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
import Learning
import torch.nn.functional as F
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
from sklearn.model_selection import StratifiedGroupKFold
import Listwise as lw
import seaborn as sns


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
    
# file_path
###########################モデルごとに変更が必要############################
field = 'nakayama3'
csv_path = f'./csv/df_all_nakayama_2025.csv'
model_type = "reg-to-reg"
# csv_path = f'./csv/df_all_{field}.csv'
###########################################################################

df = load_csv(csv_path)
group_col = 'レースID'
target_col = 'is_win'
feature_cols = []
df['is_win'] = (df['着順'] == 1).astype(int)

if __name__ == '__main__':
    seed = 1
    lw.set_seed(seed)  # 先に乱数固定
    fold_results = []

    # === 5. KFold処理 ===
    
    # ラベル作成
    df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())
    df['人気'] = df['人気'].fillna(df['人気'].median())
    # df['smooth_rel'] = lw.make_smooth_relevance_labels(df)
    # df = make_rank_labels(df)

    # 出走頭数ビン
    # bins_horses = [0, 13, 16, 100]
    # labels_horses = ['small', 'medium', 'large']
    # df['num_horses_bin'] = pd.cut(df['出走頭数'], bins=bins_horses, labels=labels_horses)

    # 反転
    df = lw.inversion(df)

    # カラム追加
    df = lw.append_col(df)
    df = lw.add_relative_features(df)

    df['label'] = 0
    df.loc[df['人気'].astype(int)==-1, 'label'] = 3
    df.loc[df['人気'].astype(int)==-2, 'label'] = 2
    df.loc[df['人気'].astype(int)==-3, 'label'] = 1
    
    print(df['label'].value_counts())

    splits, df_test_2025, df = lw.time_series_group_cv_3split_2025(df, group_col="レースID", n_splits=5)

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        # if fold == 1:
        #     break
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]
        print("fold_num:", len(train_df))
        print("fold_num:", len(val_df))
        print("fold_num:", len(test_df))
        print("fold_num:", len(df_test_2025))

        # === 6. 特徴量エンコーディング ===
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        df_2025  = df_test_2025.reset_index(drop=True)

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


        feature_cols = [col for col in train_df.columns if col not in ['label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

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
        train_df, val_df, test_df, df_2025 = lw.fill_nan(train_df, feature_cols), lw.fill_nan(val_df, feature_cols), lw.fill_nan(test_df, feature_cols), lw.fill_nan(df_2025, feature_cols)
        # カテゴリ変換
        
        train_df, map_dict = lw.race_feature_train(train_df)
        val_df = lw.race_feature_test(val_df, map_dict)
        test_df = lw.race_feature_test(test_df, map_dict)
        df_2025 = lw.race_feature_test(df_2025, map_dict)

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
        joblib.dump(map_dict, f"./pickle-dict/category_mappings_{field}_fold{fold}.pkl")

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
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            'learning_rate': rate,
            'random_state': seed,
            'verbose_eval': 1000,
            # 'objective': 'lambdarank',
            # 'metric': 'ndcg',
            # 'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
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
            # categorical_feature=cat_list,
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

        # print("test len",len(test_df))
        val_df = add_score_diff_features(val_df).round(10)
        test_df = add_score_diff_features(test_df).round(10)
        df_2025 = add_score_diff_features(df_2025).round(10)
        # print("test len",len(test_df))

        feature_cols = [col for col in val_df.columns if col not in ['label', 'label_gain', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
        joblib.dump(feature_cols,"./pickle-dict/lgb_cols_second.pkl")
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

        lgb_train = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], group=eval_list)
        lgb_eval = lgb.Dataset(test_df[feature_cols], label=test_df[target_col], reference=lgb_train, group=test_list)

        for seed in range(1, 6): 
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'regression',  # ←ここでランキング学習と指定！
                'metric': 'rmse',   # for lambdarank
                'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
                'learning_rate': rate,
                'random_state': seed,
                'verbose_eval': 1000,
                # 'objective': 'lambdarank',
                # 'metric': 'ndcg',
                # 'ndcg_eval_at': [1,3],  # NDCG@1, @3, @5, @10 を同時に計算
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
                # categorical_feature=cat_list,
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
            with open(f"./model/{field}_second_model_lgb_{model_type}_seed{seed}_{fold}.pickle", "wb") as mk:
                pickle.dump(model, mk)

            # 学習後のモデル
            importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
            importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

            feature_names = val_df[feature_cols].columns

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
            y_pred = model.predict(test_df[feature_cols])
            test_df[f'result{seed}'] = y_pred
            # print("test len",len(test_df))

            # テストデータの予測 (予測クラスを返す)
            y_pred = model.predict(df_2025[feature_cols])
            df_2025[f'result{seed}'] = y_pred


####### 評価 ########
        # テストデータの予測 (予測クラスを返す)
        # print("test len",len(test_df))
        # y_pred = model.predict(test_df[feature_cols])
        # test_df['pred_score'] = y_pred
        # # print("test len",len(test_df))

        # # テストデータの予測 (予測クラスを返す)
        # y_pred = model.predict(df_2025[feature_cols])
        # df_2025['pred_score'] = y_pred
        test_df['pred_score_second'] = test_df[['result1', 'result2', 'result3', 'result4', 'result5']].mean(axis=1).round(10)
        # print("test len",len(test_df))

        # テストデータの予測 (予測クラスを返す)
        df_2025['pred_score_second'] = df_2025[['result1', 'result2', 'result3', 'result4', 'result5']].mean(axis=1).round(10)

        test_df['expected_value'] = test_df['pred_score_second'] * test_df['オッズ']
        selected = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]
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

        # # 1. 各レースで予想順位を付ける（スコアが高いほど1位）
        # df_2025['pred_rank'] = df_2025.groupby('レースID')['pred_score'] \
        #                             .rank(ascending=False, method='first')

        # # 2. 各レースで上位3頭を抽出
        # top3 = df_2025[df_2025['pred_rank'] <= 3].copy()

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

        print(f"\n[top評価結果2025]")
        print(f"レース数: {len(top)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(top):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

        top = test_df.loc[test_df.groupby('レースID')['pred_score_second'].idxmax()]

        # 1. 各レースで予想順位を付ける（スコアが高いほど1位）
        # test_df['pred_rank'] = test_df.groupby('レースID')['pred_score'] \
        #                             .rank(ascending=False, method='first')

        # # 2. 各レースで上位3頭を抽出
        # top3 = test_df[test_df['pred_rank'] <= 3].copy()

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

        val_df.to_csv(f'./csv/{field}_result_lgb_{model_type}_val_{fold}.csv', index=False)
        test_df.to_csv(f'./csv/{field}_result_lgb_{model_type}_test_{fold}.csv', index=False)
        df_2025.to_csv(f'./csv/{field}_result_lgb_{model_type}_2025_{fold}.csv', index=False)