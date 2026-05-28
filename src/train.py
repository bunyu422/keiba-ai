"""
モデル学習・評価
- first_train : LambdaRank による1段階目の学習
- stacking    : スタッキングによる2段階目の学習
- test        : 回収率・的中率の評価
"""

import gc
import pickle

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupKFold


def first_train(df, feature_cols, cat_list, gain_list, file_num=1):
    """
    LambdaRank で1段階目の学習を行い、テストデータの予測スコアを返す。

    Parameters
    ----------
    df : pd.DataFrame
        Listwise 加工済みの DataFrame
    feature_cols : list
        学習に使用する特徴量列名のリスト
    cat_list : list
        カテゴリ列名のリスト
    gain_list : list
        LambdaRank 用 label_gain
    file_num : int
        学習する seed 数（アンサンブル数）

    Returns
    -------
    y_test_id : pd.DataFrame
        テストデータの予測スコアと正解情報
    test_list : list
        テストデータのクエリリスト（レースごとの頭数）
    """
    TRAIN_END = 202000000000
    EVAL_START = 201900000000
    TEST_END   = 202200000000

    df_train = df[df['レースID'] < EVAL_START]
    df_eval  = df[(df['レースID'] >= EVAL_START) & (df['レースID'] < TRAIN_END)]
    df_test  = df[(df['レースID'] >= TRAIN_END) & (df['レースID'] < TEST_END)]

    train_list = _query_list(df_train)
    eval_list  = _query_list(df_eval)
    test_list  = _query_list(df_test)

    y_test_id = df_test[['レースID', 'rank', 'オッズ', '着順', '単勝オッズ', '馬単']].copy()
    y_test_id['rank'] = df_test['rank']

    base_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbose': -1,
        'ndcg_eval_at': [1, 3, 5, 10, 18],
        'label_gain': gain_list,
        'learning_rate': 0.01,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'n_estimators': 10000,
    }

    for seed in range(1, file_num + 1):
        params = {**base_params, 'random_state': seed}

        lgb_train = lgb.Dataset(df_train[feature_cols], label=df_train['rank'], group=train_list)
        lgb_eval  = lgb.Dataset(df_eval[feature_cols],  label=df_eval['rank'],  reference=lgb_train, group=eval_list)

        # Optuna によるハイパーパラメータ探索
        tuner = lgb.LightGBMTunerCV(
            params, lgb_train,
            folds=GroupKFold(n_splits=3),
            categorical_feature=cat_list,
            return_cvbooster=True,
            verbose_eval=False,
        )
        tuner.run()
        best_params = tuner.best_params
        print("Best params:", best_params)
        print("Best score:", tuner.best_score)

        params = {**params, **best_params}

        # データセット再構築（メモリ解放）
        del lgb_train, lgb_eval
        gc.collect()
        lgb_train = lgb.Dataset(df_train[feature_cols], label=df_train['rank'], group=train_list)
        lgb_eval  = lgb.Dataset(df_eval[feature_cols],  label=df_eval['rank'],  reference=lgb_train, group=eval_list)

        model = lgbm.train(
            params,
            lgb_train,
            valid_names=['train', 'valid'],
            valid_sets=[lgb_train, lgb_eval],
            categorical_feature=cat_list,
            callbacks=[
                lgbm.early_stopping(stopping_rounds=20, verbose=False),
                lgbm.record_evaluation({}),
            ],
        )

        _plot_importance(model, df_train[feature_cols].columns, seed)

        y_pred = model.predict(df_test[feature_cols], group=test_list)
        y_test_id[f'result{seed}'] = y_pred

    print(f"test_list length: {len(test_list)}")
    return y_test_id, test_list


def stacking(y_test_id, gain_list, file_num=1):
    """
    1段階目の予測スコアを特徴量としてスタッキング学習を行う。

    Parameters
    ----------
    y_test_id : pd.DataFrame
        first_train の戻り値
    gain_list : list
        LambdaRank 用 label_gain
    file_num : int
        学習する seed 数
    """
    Z = _prepare_stacking_features(y_test_id, file_num)

    Z_train = Z.iloc[:int(len(Z) / 5), :]
    Z_test  = Z.iloc[int(len(Z) / 5):, :]

    train2_list = _query_list(Z_train)
    test2_list  = _query_list(Z_test)

    w_train = Z_train['rank']
    w_test  = Z_test['rank']
    Z_train = Z_train.drop(['レースID', 'rank'], axis=1)
    Z_test  = Z_test.drop(['レースID', 'rank'], axis=1)

    base_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbose': -1,
        'ndcg_eval_at': [1],
        'label_gain': gain_list,
        'learning_rate': 0.01,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'n_estimators': 10000,
    }

    for seed in range(1, file_num + 1):
        params = {**base_params, 'random_state': seed}

        zlgb_train = lgb.Dataset(Z_train, label=w_train, group=train2_list)
        zlgb_test  = lgb.Dataset(Z_test,  label=w_test,  reference=zlgb_train, group=test2_list)

        tuner = lgb.LightGBMTunerCV(
            params, zlgb_train,
            folds=GroupKFold(n_splits=3),
            return_cvbooster=True,
            verbose_eval=False,
        )
        tuner.run()
        best_params = tuner.best_params
        params = {**params, **best_params}

        del zlgb_train, zlgb_test
        gc.collect()
        zlgb_train = lgb.Dataset(Z_train, label=w_train, group=train2_list)
        zlgb_test  = lgb.Dataset(Z_test,  label=w_test,  reference=zlgb_train, group=test2_list)

        lgbm.train(
            params,
            zlgb_train,
            valid_names=['train', 'valid'],
            valid_sets=[zlgb_train, zlgb_test],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=20, verbose=False),
                lgbm.record_evaluation({}),
            ],
        )


def test(test_list, result, file_num):
    """
    回収率・的中率を計算して標準出力に表示する。

    Parameters
    ----------
    test_list : list
        クエリリスト（レースごとの頭数）
    result : pd.DataFrame
        first_train の y_test_id
    file_num : int
        seed 数
    """
    result_list = []

    for i in range(1, file_num + 1):
        result = _add_softmax_score(result, test_list, i)
        result = _sort_by_score(result, i)

        total_return = 0
        hit = 0
        count = 0
        cursor = 0

        for v in test_list:
            payout = result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float)
            total_return += payout
            if payout > 0:
                hit += 1
            count += 1
            cursor += v

        roi = total_return / (count * 100)
        result_list.append(roi)
        print(f'回収率: {roi:.2%}')
        print(f'的中率: {hit / count:.2%}')

    print(f"\n平均回収率: {np.mean(result_list):.2%} ± {np.std(result_list):.2%}")


# ------------------------------------------------------------------ #
# 内部ヘルパー
# ------------------------------------------------------------------ #

def _query_list(df):
    """レースIDごとの頭数リストを返す。"""
    return df['レースID'].value_counts(sort=False).values.tolist()


def _plot_importance(model, feature_names, seed):
    """特徴量重要度をプロットする。"""
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': model.feature_importance(importance_type='gain'),
        'importance_split': model.feature_importance(importance_type='split'),
    }).sort_values('importance_gain', ascending=False)

    print(feat_imp_df)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance_gain', y='feature', data=feat_imp_df)
    plt.title(f'Feature Importance - seed {seed}')
    plt.tight_layout()
    plt.show()


def _prepare_stacking_features(y_test_id, file_num):
    """スタッキング用に予測スコアを正規化・集計する。"""
    Z = y_test_id.copy()
    result_cols = [f'result{k}' for k in range(1, file_num + 1)]

    Z['Average'] = Z[result_cols].mean(axis=1)
    Z = Z.sort_values(['レースID', 'Average'], ascending=[True, False])

    # レース内で標準化
    id_list = Z['レースID'].value_counts(sort=False).values.tolist()
    cursor = 0
    for length in id_list[:-1]:
        for col in result_cols:
            subset = Z.iloc[cursor:cursor+length][col]
            mean, std = subset.mean(), subset.std()
            Z.iloc[cursor:cursor+length, Z.columns.get_loc(col)] = (subset - mean) / std
        cursor += length

    Z['Average'] = Z[result_cols].mean(axis=1)
    Z = Z.sort_values(['レースID', 'Average'], ascending=[True, False])

    # lower_diff / upper_diff の追加
    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    cursor = 0
    for length in id_list:
        Z.iat[cursor + length - 1, Z.columns.get_loc('lower_diff')] = float('nan')
        Z.iat[cursor, Z.columns.get_loc('upper_diff')] = float('nan')
        cursor += length

    return Z.sample(frac=1).sort_values('レースID', ascending=True)


def _add_softmax_score(result, test_list, i):
    """Softmax スコアと期待値列を追加する。"""
    cursor = 0
    for k in test_list:
        col_idx = result.columns.get_loc(f'result{i}')
        subset = result.iloc[cursor:cursor+k, col_idx].values
        exp_scores = np.exp(subset - np.max(subset))
        softmax_vals = exp_scores / exp_scores.sum()
        result[f'softmax{i}'] = np.nan
        result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')] = softmax_vals
        cursor += k

    result[f'score{i}'] = result['オッズ'] * result[f'softmax{i}']
    return result


def _sort_by_score(result, i):
    """スコア降順でソートする。"""
    return result.sort_values(['レースID', f'score{i}'], ascending=[True, False]).reset_index(drop=True)
