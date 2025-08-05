import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import Learning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
import optuna.integration.lightgbm as lgbm
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['font.family'] = 'Meiryo'


def softmax(x):
        x = x - np.max(x)  # 数値安定化
        e_x = np.exp(x)
        return e_x / e_x.sum()

def save_model(path, model):
    # モデルを保存する
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_csv(path):
    # 学習データを読み込む
    df = pd.read_csv(path, index_col=0)
    return df

def load_model(path):
    # モデルを読み込む
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(path, X_test, y_test_id, test_list):
    for i in range(1, 11):
        model = load_model(f'./pickle-tuner/{path}_{i}.pickle')
        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_test_id[f'result{i}'] = model.predict(X_test, group=test_list)
        
    return y_test_id

def parse_data(df_all):
    df_all = df_all.copy()
    SplitYear = 202000000000
    eval_year = 201900000000
    df_test = df_all[df_all['レースID'] >= SplitYear].copy()
    df_test = df_test[df_test['レースID'] < 202200000000]
    df_all = df_all[df_all['レースID'] < SplitYear]
    df_eval = df_all[df_all['レースID'] >= eval_year].copy()
    df_all = df_all[df_all['レースID'] < eval_year]

    # 説明変数,目的変数
    # '人気', 'オッズ'
    X_train = df_all.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    X_test = df_test.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    y_train = df_all[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]
    y_test = df_test[['rank', 'レースID', 'オッズ', '着順', '単勝オッズ', '馬単']]

    # train, eval, testに分割
    X_eval = df_eval.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    y_eval = df_eval[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]

    # クエリListを作成
    id_count = X_train['レースID'].value_counts(sort=False)
    train_list = id_count.values.tolist()

    id_count = X_test['レースID'].value_counts(sort=False)
    test_list = id_count.values.tolist()

    id_count = X_eval['レースID'].value_counts(sort=False)
    eval_list = id_count.values.tolist()

    # 検証用のレースIDを保存
    y_test_id = pd.DataFrame()
    y_test_id['レースID'] = y_test['レースID']
    y_test_id['着順'] = y_test['着順']
    y_test_id['単勝オッズ'] = y_test['単勝オッズ']
    y_test_id['オッズ'] = y_test['オッズ']
    y_test_id['馬単'] = y_test['馬単'].astype(int)
    y_test_id['rank'] = y_test['rank']

    # レースIDカラムを削除
    X_train = X_train.drop(['レースID'], axis=1)
    X_test = X_test.drop(['レースID'], axis=1)
    X_eval = X_eval.drop(['レースID'], axis=1)
    y_train = y_train.drop(['レースID', '着順', '単勝オッズ', '馬単'], axis=1)
    y_test = y_test.drop(['レースID', '着順', '単勝オッズ', '馬単', 'オッズ'], axis=1)
    y_eval = y_eval.drop(['レースID', '着順', '単勝オッズ', '馬単'], axis=1)

    return X_test, y_test, y_test_id, test_list

# 閾値
def threshold_eval(result, threshold1, threshold2, num):
    df_eval = result[result[f'score{num}'] > threshold1]
    df_eval = df_eval[df_eval[f'softmax{num}'] > threshold2]
    
    sum = df_eval['単勝オッズ'].sum()
    hit = len(df_eval[df_eval[f'単勝オッズ'] > 0])
    # dfの行数取得
    count = len(df_eval)
    print(f'購入数：{count}')

    return sum, hit, count

# train
def train_lgb(lgb_train, lgb_val, gain_list):
    seed = 42
    rate = 0.01
    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',  # ←ここでランキング学習と指定！
            'metric': 'ndcg',   # for lambdarank
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            'verbosity': -1,
            'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
            'label_gain': gain_list,
            'learning_rate': rate,
            'random_state': seed,
            'verbose_eval': 20,
            'early_stopping_round': 20,
            'n_estimators': 10000,
        }
    ####################################################################################

    # '''
    # クロスバリデーションによるハイパーパラメータの探索 3fold
    tuner = lgbm.LightGBMTunerCV(params,
                                lgb_train,
                                folds=GroupKFold(n_splits=3),
                                # categorical_feature = cat_list,
                                return_cvbooster=True,
                                verbose_eval=False
                                )

    # # ハイパーパラメータ探索の実行
    tuner.run()

    # # サーチしたパラメータの表示
    best_params = tuner.best_params
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    print(tuner.best_score)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': rate,
        'random_state': seed,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'n_estimators': 10000,
        'feature_pre_filter': best_params['feature_pre_filter'],
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq':  best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples'],
    }

    evals_result = {}
    model = lgb.train(params,
                    lgb_train,  # トレーニングデータの指定
                    # categorical_feature = cat_list,
                    valid_sets=[lgb_train, lgb_val],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False),
                                lgb.record_evaluation(evals_result)]
                    )

    return model

# 回帰
def regression_eval(df):
    # パラメータ設定
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 特徴量と補助変数
    feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob']]
    df['score'] = np.nan

    # ステップ1: LambdaRankでOOFスコアを作成
    # df, label_gain = Learning.create_label_gain(df)

    # ステップ2: 回帰モデルでOOF勝率を推定
    df['win_flag'] = (df['着順'] == 1).astype(int)
    df['win_prob'] = np.nan

    for fold, (train_idx, val_idx) in enumerate(kf.split(df['レースID'].unique())):
        # ランクをラベル化
        train_races = df['レースID'].unique()[train_idx]
        val_races = df['レースID'].unique()[val_idx]

        # データセット分割
        train_data = df[df['レースID'].isin(train_races)]
        val_data = df[df['レースID'].isin(val_races)]

        # グループサイズ
        # group_train = train_data.groupby('レースID').size().to_list()
        # group_val = val_data.groupby('レースID').size().to_list()

        # データセット作成
        # lgb_train = lgb.Dataset(train_data[feature_cols], label=train_data['rank'], group=group_train)
        # lgb_val = lgb.Dataset(val_data[feature_cols], label=val_data['rank'], group=group_val, reference=lgb_train)

        # モデル定義
        # model = lgb.LGBMClassifier(
        #     objective='binary',
        #     metric='binary_logloss',
        #     n_estimators=100,
        #     random_state=42
        # )

        # # モデル学習
        # model.fit(train_data[feature_cols], train_data['win_flag'])

        # # モデルを保存
        # save_model(f'./pickle-tuner/tokyo_reg_{fold}.pkl', model)

        with open(f'./pickle-tuner/tokyo_reg_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)

        # OOF予測
        df.loc[val_data.index, 'score'] = model.predict_proba(val_data[feature_cols])[:, 1]


    # df['score'] = df.groupby('レースID')['score'].rank(ascending=False, method='first')

    # 各レースでsoftmax
    df['score'] = df.groupby('レースID')['score'].transform(softmax)

    # 期待値と評価
    df['expected_value'] = df['score'] * df['オッズ']
    buy_signals = df.loc[df.groupby('レースID')['expected_value'].idxmax()]
    # buy_signals = buy_signals[buy_signals['expected_value'] >= 1.5]
    hits = (buy_signals['着順'] == 1).sum()
    roi = buy_signals['単勝オッズ'].sum() / (len(buy_signals) * 100) if len(buy_signals) > 0 else 0

    print(f"買い対象: {len(buy_signals)}頭")
    print(f"的中数: {hits}")
    print(f"回収率: {roi:.2f}倍")
    print(f"的中率： {hits / len(buy_signals):.2f}")

    # KFold
    kf2 = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf2.split(df.dropna(subset=['score']))):
        # 分割
        train_data = df.iloc[train_idx].dropna(subset=['score'])
        val_data = df.iloc[val_idx].dropna(subset=['score'])

        # Isotonic Regression（スコアと勝率を単調増加でフィッティング）
        iso = IsotonicRegression(increasing=False,out_of_bounds='clip')  # 範囲外スコアにも対応
        iso.fit(train_data['score'], train_data['win_flag'])

        with open(f'./pickle-tuner/tokyo_iso_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)

        df.loc[val_data.index, 'win_prob'] = iso.predict(val_data['score'])

        # モデル保存
        # save_model(f'./pickle-tuner/tokyo_iso_{fold}.pkl', iso)

    # 各レースでsoftmax
    df['win_prob'] = df.groupby('レースID')['win_prob'].transform(softmax)

    # 期待値と評価
    df['expected_value'] = df['win_prob'] * df['オッズ']
    buy_signals = df.loc[df.groupby('レースID')['expected_value'].idxmax()]
    # buy_signals = buy_signals[buy_signals['expected_value'] >= 1.5]
    hits = (buy_signals['着順'] == 1).sum()
    roi = buy_signals['単勝オッズ'].sum() / (len(buy_signals) * 100) if len(buy_signals) > 0 else 0

    print(f"買い対象: {len(buy_signals)}頭")
    print(f"的中数: {hits}")
    print(f"回収率: {roi:.2f}倍")
    print(f"的中率： {hits / len(buy_signals):.2f}")

    return df

def eval(df):
    # パラメータ設定
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 特徴量と補助変数
    feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob']]
    df['score'] = np.nan

    for fold, (train_idx, val_idx) in enumerate(kf.split(df['レースID'].unique())):
        # ランクをラベル化
        train_races = df['レースID'].unique()[train_idx]
        val_races = df['レースID'].unique()[val_idx]

        # データセット分割
        train_data = df[df['レースID'].isin(train_races)]
        val_data = df[df['レースID'].isin(val_races)]

        # モデル読み込み
        with open(f'./pickle-tuner/tokyo_rank_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)

        # OOF予測
        df.loc[val_data.index, 'score'] = model.predict(val_data[feature_cols], num_iteration=model.best_iteration)

    # ステップ2: 回帰モデルでOOF勝率を推定
    df['win_flag'] = (df['着順'] == 1).astype(int)
    df['win_prob'] = np.nan

    # 各レースでsoftmax
    df['score'] = df.groupby('レースID')['score'].transform(softmax)

    # スコアで順位付け
    df['pre_rank'] = df.groupby('レースID')['score'].rank(ascending=True, method='first')

    # 順位を反転してスケーリング    
    df['scaled_rank'] = df['pre_rank'] / (df['pre_rank'].max() - 1)
    df['score_x_rank'] = df['score'] * df['scaled_rank']

    df['pre_rank'] = df['pre_rank'].astype('category')

    train_col = ['score_x_rank', 'pre_rank', 'score']
    target_col = 'win_flag'

    # スコアを全体でビンに分割（例：10ビン）
    df = df.loc[df.groupby('レースID')['score'].idxmax()]
    # df['score_bin'] = pd.qcut(df['score'], q=10, labels=False, duplicates='drop')

    # # 各ビンごとの平均スコアと勝率を計算
    # bin_stats = df.groupby('score_bin').agg(
    #     mean_score=('score', 'mean'),
    #     win_rate=('win_flag', 'mean'),
    #     count=('win_flag', 'count')
    # ).reset_index()

    # # 勝率プロット
    # plt.figure(figsize=(8, 5))
    # plt.plot(bin_stats['mean_score'], bin_stats['win_rate'], marker='o')
    # plt.xlabel('平均スコア（ビンごと）')
    # plt.ylabel('1着率（勝率）')
    # plt.title('スコアと勝率の関係')
    # plt.grid(True)
    # plt.show()

    # スコアの範囲を 0〜1 に正規化（必要ならレース単位）
    # df['score'] = df.groupby('レースID')['score'].transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1,1)).flatten())

    # KFold
    kf2 = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf2.split(df.dropna(subset=['score']))):
        # 分割
        train_data = df.iloc[train_idx].dropna(subset=['score'])
        val_data = df.iloc[val_idx].dropna(subset=['score'])

        # print(train_data[train_col].describe())
        # print(train_data['win_flag'].value_counts())
        # print(train_data[train_col].corr())  # 相関


        # # Isotonic Regression（スコアと勝率を単調増加でフィッティング）
        # iso = IsotonicRegression(increasing=False,out_of_bounds='clip')  # 範囲外スコアにも対応
        # iso.fit(train_data['score'], train_data['win_flag'])

        clf = LogisticRegression()
        clf.fit(train_data['score'].values.reshape(-1, 1), train_data['win_flag'])

        # calibrated_probs = clf.predict_proba(X.reshape(-1, 1))[:, 1]

        

        # # モデル読み込み
        # with open(f'./pickle-tuner/tokyo_iso_{fold}.pkl', 'rb') as f:
        #     iso = pickle.load(f)

        # model = lgb.LGBMRegressor(
        #     objective='regression',
        #     learning_rate=0.05,
        #     n_estimators=100,
        #     max_depth=4,
        #     random_state=42
        # )
        # model.fit(train_data[train_col], train_data[target_col])

        # # 予測
        df.loc[val_data.index, 'win_prob'] = clf.predict_proba(val_data['score'].values.reshape(-1, 1))[:, 1]
        # df.loc[val_data.index, 'win_prob'] = iso.predict(val_data['score'])
        # df.loc[val_data.index, 'win_prob'] = iso.predict(val_data[train_col])

        # # # モデル保存
        # save_model(f'./pickle-tuner/tokyo_reg_{fold}.pkl', model)
    
    

    # 期待値と評価

    # log変換とシャープレシオ的補正を組み合わせた期待値
    # df['log_odds'] = np.log(df['オッズ'] + 1)
    # df['expected_value'] = df['win_prob'] * df['log_odds'] / (1 + df['log_odds'])

    df['expected_value'] = df['win_prob'] * df['オッズ']
    buy_signals = df[df['expected_value'] >= 1.2]
    # 各レースで期待値最大の馬を1頭だけ抽出
    # best_horse_per_race = df.loc[df.groupby('レースID')['expected_value'].idxmax()]

    # 的中数（着順が1なら的中）
    # hits = (best_horse_per_race['着順'] == 1).sum()

    # # 回収率の計算
    # roi = best_horse_per_race.loc[best_horse_per_race['着順'] == 1, '単勝オッズ'].sum() / (len(best_horse_per_race) * 100)

    # # 的中率の計算（任意）
    # hit_rate = hits / len(best_horse_per_race)

    # # 結果表示
    # print(f"対象レース数: {len(best_horse_per_race)}")
    # print(f"的中数: {hits}")
    # print(f"的中率: {hit_rate:.2%}")
    # print(f"回収率: {roi:.2f}倍")

    hits = (buy_signals['着順'] == 1).sum()
    roi = buy_signals['単勝オッズ'].sum() / (len(buy_signals) * 100) if len(buy_signals) > 0 else 0

    print(f"買い対象: {len(buy_signals)}頭")
    print(f"的中数: {hits}")
    print(f"回収率: {roi:.2f}倍")
    print(f"的中率： {hits / len(buy_signals):.2f}")

    return df

def test(test_list, result, file_num):
    result_list = []
    for i in range(1, file_num+1):
        sum = 0
        hit = 0
        count = 0
        cursor = 0
        result[f'softmax{i}'] = np.nan

        for k in test_list:
            # スコア抽出
            subset = result.iloc[cursor:cursor+k, result.columns.get_loc(f'result{i}')].values

            temperature = 1.0  # 差を強調したいなら1.0未満、緩和したいなら1.0超

            exp_scores = np.exp((subset - np.max(subset)) / temperature)
            softmax_values = exp_scores / exp_scores.sum()

            # print(softmax_values)

            # print("softmax max:", softmax_values.max())
            # print("softmax min:", softmax_values.min())

            # 結果を DataFrame に格納（例として新しい列に）
            result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')] = softmax_values
            # print(result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')])
            cursor += k

        # 期待値
        result[f'score{i}'] = result['オッズ'] * result[f'softmax{i}']
        cursor = 0
        result = Learning.sort(result, i)
        for v in test_list:
            sum += result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float)
            if result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float) > 0:
                hit += 1
            count += 1
            cursor += v

        # sum, hit, count = threshold_eval(result, 0.0, 0.2, i)
        result_list.append(sum / (count * 100))
        print(f'回収率:{sum / (count * 100)}')
        print(f'的中率:{hit / count}')
    
    mean_return = np.mean(result_list)
    std_return = np.std(result_list)

    print(f"平均回収率: {mean_return:.2%} ± {std_return:.2%}")

def softmax_test(test_list, result, file_num):
    result_list = []
    for i in range(1, file_num+1):
        sum = 0
        hit = 0
        count = 0
        cursor = 0
        result[f'softmax{i}'] = np.nan

        for k in test_list:
            # スコア抽出
            subset = result.iloc[cursor:cursor+k, result.columns.get_loc(f'result{i}')].values

            temperature = 0.5  # 差を強調したいなら1.0未満、緩和したいなら1.0超

            exp_scores = np.exp((subset - np.max(subset)) / temperature)
            softmax_values = exp_scores / exp_scores.sum()

            # print(softmax_values)

            # print("softmax max:", softmax_values.max())
            # print("softmax min:", softmax_values.min())

            # 結果を DataFrame に格納（例として新しい列に）
            result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')] = softmax_values
            # print(result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')])
            cursor += k

        # 期待値
        # 3. スコアのbinを作成（0.01刻みで丸める）
        result['score_bin'] = result[f'softmax{i}'].apply(lambda x: round(x, 2))

        # 4. 勝者フラグ（1着かどうか）
        result['is_winner'] = (result['着順'] == 1).astype(int)

        # 5. 各binごとに勝率計算
        winrate_by_bin = result.groupby('score_bin')['is_winner'].mean().to_dict()
        print(winrate_by_bin)

        # 4. 各馬に「自身のスコアビンの勝率」を付与
        result['score_winrate'] = result['score_bin'].map(winrate_by_bin)

        # 6. 結果を出力
        result[f'score{i}'] = result['オッズ'] * result[f'score_winrate']

        best_horse_per_race = result.loc[result.groupby('レースID')['score_winrate'].idxmax()]
        print(f'レース数：{len(best_horse_per_race)}')

        best_horse_per_race = best_horse_per_race[best_horse_per_race[f'score{i}'] > 1.0]

        # 的中数（着順が1なら的中）
        hits = (best_horse_per_race['着順'] == 1).sum()

        # 回収率の計算
        roi = best_horse_per_race.loc[best_horse_per_race['着順'] == 1, '単勝オッズ'].sum() / (len(best_horse_per_race) * 100)

        # 的中率の計算（任意）
        hit_rate = hits / len(best_horse_per_race)

        # 結果表示
        print(f"対象レース数: {len(best_horse_per_race)}")
        print(f"的中数: {hits}")
        print(f"的中率: {hit_rate:.2%}")
        print(f"回収率: {roi:.2f}倍")

        # cursor = 0
        # result = Learning.sort(result, i)
        # for v in test_list:
        #     sum += result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float)
        #     if result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float) > 0:
        #         hit += 1
        #     count += 1
        #     cursor += v

        # sum, hit, count = threshold_eval(result, 0.0, 0.2, i)
        result_list.append(roi)
        # print(f'回収率:{sum / (count * 100)}')
        # print(f'的中率:{hit / count}')
    
    mean_return = np.mean(result_list)
    std_return = np.std(result_list)

    print(f"平均回収率: {mean_return:.2%} ± {std_return:.2%}")

# ==== ペアデータ生成関数 ====
def create_pair_data(df_subset):
    pair_rows = []
    for race_id, group in df_subset.groupby('レースID'):
        horses = group.reset_index()
        for i in range(len(horses)):
            for j in range(i + 1, len(horses)):
                h1 = horses.loc[i]
                h2 = horses.loc[j]
                ev1 = h1['expected_value']
                ev2 = h2['expected_value']
                if ev1 == ev2:
                    continue
                label = 1 if ev1 > ev2 else 0
                feature_diff = h1[feature_cols].values - h2[feature_cols].values
                common_features = h1[common_cols].values
                full_features = np.concatenate([feature_diff, common_features])
                pair_rows.append((full_features, label))
    if not pair_rows:
        return np.array([]), np.array([])
    X = np.array([row[0] for row in pair_rows])
    y = np.array([row[1] for row in pair_rows])
    return X, y

# ==== 予測関数 ====
def predict_dueling(df_subset, model):
    results = []
    for race_id, group in df_subset.groupby('レースID'):
        group = group.copy().reset_index()
        scores = np.zeros(len(group))
        common_features = group.loc[0, common_cols].values
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue
                diff = group.loc[i, feature_cols].values - group.loc[j, feature_cols].values
                full_input = np.concatenate([diff, common_features])
                prob = model.predict_proba([full_input])[0][1]
                scores[i] += prob
        best_idx = np.argmax(scores)
        results.append(group.loc[best_idx])
    return pd.DataFrame(results)

if __name__ == '__main__':
    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # file_path
    csv_path = './csv/df_all.csv'
    model_path = 'tokyotest'

    feature_cols = ['斤量', '人気', '父馬', 
    '1過去着順',  '1タイム',  '1後3F',
    '1着差', '1スピード指数',
    '2過去着順',  '2タイム', '2後3F',
    '2着差', '2スピード指数',
    '3過去着順', '3タイム', '3後3F',
    '3着差', '3スピード指数', 
    '4過去着順', '4タイム', '4後3F',
    '4着差', '4スピード指数',
    '5過去着順', '5タイム', '5後3F',
    '5着差', '5スピード指数', 
    'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度']

    common_cols = ['距離', 'フィールド', '馬場', '出走頭数', '間隔', '性', '齢', '1距離', '1馬場', '1斤量', '1コーナー通過順',
                   '1馬体重', '1体重増減', '1クラス', '1距離差', '1場所変化', '1フィールド変化',
                   '2距離', '2馬場', '2斤量', '2コーナー通過順', '2馬体重', '2体重増減', '2クラス', '2距離差', '2場所変化', '2フィールド変化','2距離',
                   '3距離', '3馬場', '3斤量', '3コーナー通過順', '3馬体重', '3体重増減', '3クラス', '3距離差', '3場所変化', '3フィールド変化',
                   '4距離', '4馬場', '4斤量', '4コーナー通過順', '4馬体重', '4体重増減', '4クラス', '4距離差', '4場所変化', '4フィールド変化',
                   '5距離', '5馬場', '5斤量', '5コーナー通過順', '5馬体重', '5体重増減', '5クラス', '5距離差', '5場所変化', '5フィールド変化',
                   '平均クラス', '平均ペース', '1クラス差', '1ペース差',]


    # テストデータを読み込む
    df = load_csv(csv_path)
    # print(df.columns.values)

    # df = eval(df)
    regression_eval(df)

    # ==== GroupKFoldによる交差検証 ====
    # gkf = GroupKFold(n_splits=5)
    # race_ids = df['レースID'].values
    # unique_race_ids = df['レースID'].unique()

    # accuracies = []
    # rois = []

    # for fold, (train_idx, test_idx) in enumerate(gkf.split(unique_race_ids, groups=unique_race_ids), 1):
    #     train_race_ids = unique_race_ids[train_idx]
    #     test_race_ids = unique_race_ids[test_idx]
        
    #     train_df = df[df['レースID'].isin(train_race_ids)]
    #     test_df = df[df['レースID'].isin(test_race_ids)]

    #     X_train, y_train = create_pair_data(train_df)
    #     if len(X_train) == 0:
    #         print(f"Fold {fold}: スキップ（学習データ不足）")
    #         continue

    #     model = lgb.LGBMClassifier(
    #                 objective='binary',  # 2クラス分類
    #                 boosting_type='gbdt',
    #                 n_estimators=100,
    #                 learning_rate=0.01,
    #                 random_state=42
    #             )
    #     model.fit(X_train, y_train)

    #     predicted = predict_dueling(test_df, model)
    #     if predicted.empty:
    #         print(f"Fold {fold}: スキップ（予測対象なし）")
    #         continue

    #     hits = (predicted['着順'] == 1).sum()
    #     roi = predicted['単勝オッズ'].sum() / (len(predicted) * 100)

    #     acc = hits / len(predicted)
    #     accuracies.append(acc)
    #     rois.append(roi)

    #     print(f"Fold {fold}: 的中率 = {acc:.3f}, 回収率 = {roi:.2f}")

    # # ==== 結果の平均表示 ====
    # print(f"\n平均的中率: {np.mean(accuracies):.3f}")
    # print(f"平均回収率: {np.mean(rois):.2f}")

    # 学習
    # regression_eval(df)

    # 評価
    # eval(df)

    # テストデータを読み込む
    # X_test, y_test, y_test_id, test_list = parse_data(df)

    # # 予測
    # y_test_id = predict(model_path ,X_test, y_test_id, test_list)

    # # 評価
    # softmax_test(test_list, y_test_id, 10)

#     ['着順' '馬番' '斤量' '騎手' '人気' '単勝オッズ' '距離' 'フィールド' '馬場' '出走頭数' '馬単' 'レースID'
#  '父馬' '間隔' '性' '齢' '1場所' '1過去着順' '1フィールド' '1距離' '1タイム' '1馬場' '1出走馬数' '1馬番'
#  '1人気' '1斤量' '1コーナー通過順' '1後3F' '1馬体重' '1体重増減' '1着差' '1クラス' '1スピード指数'
#  '1距離差' '1場所変化' '1フィールド変化' '2場所' '2過去着順' '2フィールド' '2距離' '2タイム' '2馬場'
#  '2出走馬数' '2馬番' '2人気' '2斤量' '2コーナー通過順' '2後3F' '2馬体重' '2体重増減' '2着差' '2クラス'
#  '2スピード指数' '2距離差' '2場所変化' '2フィールド変化' '3場所' '3過去着順' '3フィールド' '3距離' '3タイム'
#  '3馬場' '3出走馬数' '3馬番' '3人気' '3斤量' '3コーナー通過順' '3後3F' '3馬体重' '3体重増減' '3着差'
#  '3クラス' '3スピード指数' '3距離差' '3場所変化' '3フィールド変化' '4場所' '4過去着順' '4フィールド' '4距離'
#  '4タイム' '4馬場' '4出走馬数' '4馬番' '4人気' '4斤量' '4コーナー通過順' '4後3F' '4馬体重' '4体重増減'
#  '4着差' '4クラス' '4スピード指数' '4距離差' '4場所変化' '4フィールド変化' '5場所' '5過去着順' '5フィールド'
#  '5距離' '5タイム' '5馬場' '5出走馬数' '5馬番' '5人気' '5斤量' '5コーナー通過順' '5後3F' '5馬体重'
#  '5体重増減' '5着差' '5クラス' '5スピード指数' '5距離差' '5場所変化' '5フィールド変化' '平均クラス' '平均ペース'
#  '1クラス差' '1ペース差' 'best着差' 'bestスピード指数' 'av着差' 'avスピード指数' '上昇度' 'オッズ'
#  'rank']