import pickle
import pandas as pd
import Learning

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

if __name__ == '__main__':
    # file_path
    csv_path = './csv/df_all.csv'
    model_path = 'tokyotest'

    # テストデータを読み込む
    df = load_csv(csv_path)

    # テストデータを読み込む
    X_test, y_test, y_test_id, test_list = parse_data(df)

    # 予測
    y_test_id = predict(model_path ,X_test, y_test_id, test_list)

    # 評価
    Learning.test(test_list, y_test_id, 10)