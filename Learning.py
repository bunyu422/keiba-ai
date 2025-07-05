import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupKFold
import pickle
from IPython.display import display
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
import time
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from keras.callbacks import EarlyStopping
import sys
import category_encoders as ce

warnings.simplefilter('ignore')

optuna.logging.disable_default_handler()

# 開催場所番号
field = 2

# {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
#  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
#  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

# 学習済みモデルを保存するファイルネーム
tuner_name = "tokyo"
file_name = 'tokyo'

# ファイル数
file_num = 5

# ファイルパス
csv_path = f"./csv/{file_name}_2012-2024.csv" # 学習に使うcsvデータのパス
horse_path = "./pickle-dict/horse_jra.pkl" # 父馬のマッピング用辞書のパス
femal_horse_path = "./pickle-dict/femal_horse_jra.pkl" # 母父馬のマッピング用辞書のバス
jockey_path = "./pickle-dict/jockey_jra.pkl" # 騎手のマッピング用辞書のパス
tuner_path = f"./pickle-tuner/{tuner_name}test_" # 学習済みモデルを保存する場所

# 目的変数作成
f_ranking = {1: 10, 2: 5, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0,
            '1': 10, '2': 5, '3': 3, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0}

nagoya_mapping = {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
                    '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
                    '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

class_dict = {'GI': 1, 'GII': 2, 'GIII': 3, 'OP': 4, 'L': 4, '3勝': 5, '1600万下': 5, '1600下': 5, '2勝': 6, '1000万下': 6, '1000下': 6,
                '1勝': 7, '500万下': 7, '500下': 7, '未勝利': 8, '新馬': 9}

field_mapping = {'芝': 1, 'ダ': 2, '障': 3}

condition_mapping = {'良': 1, '稍': 2, '重': 3, '不': 4}

base_time = pd.to_datetime('00:00.0', format='%M:%S.%f')

# label_gain設定用のリスト
gain_list = [int(i) for i in range(1,30)]

# 関数
def target_encording(df, column, target):
    tem = pd.DataFrame()
    df_tem = pd.DataFrame()
    df_ind = pd.DataFrame()
    dfs = [df.iloc[i:i+int(len(df.index)/5)+1, :] for i in range(0, len(df.index), int(len(df.index) / 5) + 1)]

    for i in range(5):
        df_tem = dfs[i].copy()
        df_ind = dfs.copy()
        del df_ind[i]
        df_ind = pd.concat([dfs[0], dfs[1], dfs[2], dfs[3]], axis=0)
        d = df_ind.groupby(column)[target].mean()
        dict = d.to_dict()
        df_tem[column] = pd.to_numeric(df_tem[column].map(dict), errors='coerce')
        tem = pd.concat([tem, df_tem], axis=0)
    
    return tem

# スクレイピング
def scraping(csv_path):
    # ヘッダー
    headers = {'User-Agent': 'Mozilla/5.0'}
    for year in range(2012, 2025):
        for number in range(1, 6):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    race_id = '{}10{}{}{}'.format(str(year), str(number).zfill(2), str(day).zfill(2), str(race_no).zfill(2))
                    url_race = 'https://race.netkeiba.com/race/result.html?race_id={}&rf=race_list'.format(race_id)
                    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
                    try:
                        response_race = requests.get(url_race, headers=headers)
                        response_past = requests.get(url_past, headers=headers)
                        df_result = pd.read_html(response_race.content)[0]
                        df_past = pd.read_html(response_past.content)[0]
                        soup = BeautifulSoup(response_race.content, 'html.parser')
                        data1 = soup.find('div', class_='RaceData01').text
                        data2 = soup.find('div', class_='RaceData02').text
                        data3 = soup.find('tr', class_='Umatan').text
                        data4 = soup.find('h1', class_='RaceName').text
                        a = data2[data2.find('新馬')+0: data2.find('新馬')+2]
                        if a == '新馬':
                            continue
                        df_result_past = pd.merge(df_result, df_past, on='馬番')
                        df_result_past['距離'] = re.findall(r'\d+', data1)[2]
                        df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
                        df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
                        df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]
                        df_result_past['馬単'] = data3
                        df_result_past['レース名'] = data4.replace('\n', '')
                        print(url_race)
                        time.sleep(1)
                    except:
                        continue
                    df_result_past['レースID'] = race_id
                    df = pd.concat([df, df_result_past])

    # 結果をcsvに保存
    df.to_csv(csv_path, na_rep='NaN')

# 辞書作成
def create_unique_pickle(series, file_path):
    mapping = dict(zip(series.unique().tolist(), range(1, len(series.unique().tolist()) + 1)))
    with open(file_path, "wb") as jd:
        pickle.dump(mapping, jd)
    
    return mapping

# 辞書作成
def return_pickle(file_path):
    with open(file_path, "rb") as jd:
        mapping = pickle.load(jd)
    
    return mapping

# データの初期加工
def df_first_processing(df):
    # 新たなカラムを作成
    df['父馬'] = df['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df['間隔'] = df['馬名_y'].str.extract(r'(\d+)', expand=True)
    df['母父馬'] = df['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    # 血統pickle作成
    femal_mapping = create_unique_pickle(df['母父馬'], femal_horse_path)
    horse_mapping = create_unique_pickle(df['父馬'], horse_path)

    
    df['rank'] = df['着順'].map(f_ranking)

    # マッピング
    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_mapping), errors='coerce')

    # 馬単の配当を抽出
    df['馬単'] = df['馬単'].str.replace(',', '')
    df['馬単'] = df['馬単'].str.extract(r'(\d+円)', expand=True)
    df['馬単'] = df['馬単'].str.extract(r'(\d+)', expand=True)

    # いらないカラムを消す
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', 'コーナー通過順', '厩舎', 'タイム', '騎手斤量', '着差', '後3F', '印'], axis=1)

    # 特殊記号を消す
    df['騎手'] = df['騎手'].str.replace('▲', '')
    df['騎手'] = df['騎手'].str.replace('△', '')
    df['騎手'] = df['騎手'].str.replace('☆', '')
    df['騎手'] = df['騎手'].str.replace('★', '')
    df['騎手'] = df['騎手'].str.replace('◇', '')

    # 1つのカラムに入っているデータを複数カラムに分ける
    df_sex = df['性齢'].str.extract(r'([牝牡セ])(\d+)', expand=True)
    df['性'] = df_sex.loc[:, 0]
    df['齢'] = df_sex.loc[:, 1]

    df_weight = df['馬体重(増減)'].str.extract(r'(\d{3}).([+-0]\d*)', expand=True)
    df['馬体重'] = df_weight.loc[:, 0]
    df['体重増減'] = df_weight.loc[:, 1].str.replace('\+', '', regex=True)

    # 性別をマッピング
    sex_mapping = {'牡': 1, '牝': 2, 'セ': 3}
    df['性'] = df['性'].map(sex_mapping)

    df['フィールド'] = df['フィールド'].map(field_mapping)
    df['馬場'] = df['馬場'].map(condition_mapping)

    # 騎手のユニーク値から辞書をつくる
    jockey_mapping = create_unique_pickle(df['騎手'], jockey_path)

    df['騎手'] = df['騎手'].map(jockey_mapping)

    df['勝率'] = 0
    df['勝率'] = df['勝率'].mask(pd.to_numeric(df['着順'], errors='coerce') == 1, 1)

    d = df.groupby('騎手')['勝率'].mean()
    dict = d.to_dict()
    df['騎手'] = pd.to_numeric(df['騎手'].map(dict), errors='coerce')

    with open(f'./pickle-dict/jwin_dict{field}.pkl', "wb") as dd:
        pickle.dump(dict, dd)

    # 「性齢」「馬体重（増減）」はいらないので消す
    df = df.drop(['性齢', '馬体重(増減)', '馬体重', '体重増減'], axis=1)

# 2~5走前のデータを処理
def df_big_past_processing(df):
    # 騎手の辞書を読み込み
    jockey_mapping = return_pickle(jockey_path)

    # 以降2~5走の処理
    for sou in range(1, 6):
        if sou == 1:
            df_split = df['前走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
        else:
            df_split = df[sou+'走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
        df_split.columns = ['日付', sou+'場所', sou+'過去着順', sou+'レース名', sou+'フィールド', sou+'距離', sou+'タイム', sou+'馬場', sou+'出走馬数', sou+'馬番', sou+'人気', sou+'騎手', sou+'斤量', sou+'コーナー通過順', sou+'後3F', sou+'馬体重', sou+'体重増減', sou+'着差']

        # 2~5走のカラムを削除
        df_split = df_split.drop(['日付'], axis=1)

        # 4角コーナー通過順のみに
        df_split[sou+'コーナー通過順'] = df_split[sou+'コーナー通過順'].str[-4:-1].apply(lambda x : x if x != ' ' else None)
        df_split[sou+'コーナー通過順'] = df_split[sou+'コーナー通過順'].astype(float).abs()

        # クラス別に分類
        df_split[sou+'クラス'] = 0
        for k, v in class_dict.items():
            df_split[sou+'クラス'] = df_split[sou+'クラス'].mask(df_split[sou+'レース名'].str.contains(k, na=False), v)


        # 文字列データを数値データにする
        df_split[sou+'場所'] = df_split[sou+'場所'].map(nagoya_mapping)

        df_split[sou+'フィールド'] = df_split[sou+'フィールド'].map(field_mapping)

        df_split[sou+'馬場'] = df_split[sou+'馬場'].map(condition_mapping)

        # 文字列から数値に変換する
        df_split[sou+'騎手'] = df_split[sou+'騎手'].map(jockey_mapping)

        # タイムを秒表記にする
        df_split[sou+'タイム'] = pd.to_datetime(df_split[sou+'タイム'], format='%M:%S.%f') - base_time
        df_split[sou+'タイム'] = df_split[sou+'タイム'].dt.total_seconds()

        # # スピード指数の計算
        for i in range(1, 11):
            for k in range(1000, 3700, 100):
                try:
                    speed = speed_dict[f'{i}{k}1']
                    df_split.loc[(df_split[sou+'場所'] == i) & (df_split[sou+'距離'].astype(float) == k) & (df_split[sou+'フィールド'] == 1), sou+'スピード指数'] = ((speed + 0.01 - df_split[sou+'タイム'].astype(float)) * 10) * (1 / speed * 100) + (df_split[sou+'馬場'] * 10) + (df_split[sou+'斤量'].astype(float) - 55) * 2 + 80
                except KeyError:
                    pass
                try:
                    speed = speed_dict[f'{i}{k}2']
                    df_split.loc[(df_split[sou+'場所'] == i) & (df_split[sou+'距離'].astype(float) == k) & (df_split[sou+'フィールド'] == 2), sou+'スピード指数'] = ((speed + 0.01 - df_split[sou+'タイム'].astype(float)) * 10) * (1 / speed * 100) + (13 - df_split[sou+'馬場'] * 3) + (df_split[sou+'斤量'].astype(float) - 55) * 2 + 80
                except KeyError:
                    pass

        # クラス別に分類
        df_split[sou+'クラス'] = 0
        for k, v in class_dict.items():
            df_split[sou+'クラス'] = df_split[sou+'クラス'].mask(df_split[sou+'レース名'].str.contains(k, na=False), v)

        # df_split[sou+'クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_split[sou+'クラス'].astype(str) + df_split[sou+'過去着順'].astype(str), errors='coerce')

        df_split[sou+'着差'] = df_split[sou+'着差'].astype(float) + (df_split[sou+'クラス'].astype(int) * 0.5)

        # 上がり3Fを指数化
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 1) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.94 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 2) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (1.01 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 3) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.36 + (df_split[sou+'距離'].astype(float) * 1.5 / 100000)))

        df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
        df_split[sou+'場所変化'] = df_split[sou+'場所'] - field
        df_split[sou+'フィールド変化'] = df_split[sou+'フィールド'] - df['フィールド']

        # 不要なカラムを削除
        # df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)
        df_split = df_split.drop([sou+'レース名', sou+'騎手'], axis=1)


        # 今走と過去走を結合
        df_all = pd.concat([df_all, df_split], axis=1)

        if sou == 1:
            past_level(df_all)

    return df_all

def past_level(df_all):
    # クエリListを作成
    id_count = df_all['レースID'].value_counts(sort=False)
    n_list = id_count.values.tolist()

    n_list = n_list[:-1]
    n = 0
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    for i in n_list:
        n += i
        df_all.iloc[n-i:n, df_all.columns.get_loc('平均クラス')] = df_all.iloc[n-i:n, df_all.columns.get_loc('1クラス')].mean().astype(int)
        df_all.iloc[n-i:n, df_all.columns.get_loc('平均ペース')] = df_all.iloc[n-i:n, df_all.columns.get_loc('1コーナー通過順')].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']

# テストデータを分離してターゲットエンコーディング
def encording(df_all):
    df_test = df_all[df_all['レースID'] >= 202000000000].copy()
    df_all = df_all[df_all['レースID'] < 202000000000]

    d = df_all.groupby('父馬')['rank'].mean()
    dict = d.to_dict()
    df_test['父馬'] = pd.to_numeric(df_test['父馬'].astype(float).map(dict), errors='coerce')
    with open(f'./pickle-dict/sire_dict{field}.pkl', "wb") as dd:
        pickle.dump(dict, dd)
    
    df_all = target_encording(df_all, '父馬', 'rank')

    df_all = pd.concat([df_all, df_test], axis=0)

# 終盤のデータ加工
def df_end_processing(df_all):
    # 着順から文字列を排除
    indexNames = df_all[(df_all['着順'] != '中止') & (df_all['着順'] != '除外') & (df_all['着順'] != '取消') & (df_all['着順'] != '失格') & (df_all['着順'] != '未定')]
    df_all = indexNames

    # 着差とスピード指数のbest, avカラム作成
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬', 'レース名', '勝率'], axis=1)

    print(df_all.isnull().sum())

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)
    df_all['上昇度'] = (df_all['5過去着順'] - df_all['4過去着順']) + (df_all['4過去着順'] - df_all['3過去着順']) + (df_all['3過去着順'] - df_all['2過去着順']) + (df_all['2過去着順'] - df_all['1過去着順']) / (df_all[['1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順']].isnull().sum(axis=1) + 1)
    
    # df_all['rank'] = df_all['着順'].map(f_ranking)
    df_all = df_all.replace('', '00000')  # ''をエラー検出文字に置換してくれる
    df_all = df_all.replace('未定', '00000')
    df_all = df_all[~df_all.apply(lambda s: s.str.contains('00000'), axis=1).any(axis=1)]  # エラー検出文字を入れた行以外を抽出
    df_all = df_all.astype(float)

    # 1着のみ単勝オッズを保有
    df_all['オッズ'] = df_all['単勝オッズ']
    df_all['単勝オッズ'] = df_all['単勝オッズ'].where(df_all['着順'].astype(int) == 1, 0)
    df_all['単勝オッズ'] = df_all['単勝オッズ'] * 100

    # レース内の順序をシャッフル
    df_all = df_all.sample(frac=1)
    df_all = df_all.sort_values(['レースID'], ascending=True)

# 1段階目の学習
def first_train(df_all):
    # 2023のデータを分離
    # 2012-2018 学習データ, 2019 評価データ, 2020-2021 stacking, 2022-2023 テストデータ
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

    # dataframeを値のみに
    display(list(X_train.columns.values))

    # パラメータ設定
    rate = 0.01

    for seed in range(1, file_num+1):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',  # ←ここでランキング学習と指定！
            'metric': 'ndcg',   # for lambdarank
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
            'label_gain': gain_list,
            'learning_rate': rate,
            'random_state': seed,
            'verbose_eval': 20,
            'early_stopping_round': 20,
            'n_estimators': 10000,
        }
        ####################################################################################

        # データセット再構築
        del lgb_train
        del lgb_eval
        gc.collect()
        lgb_train = lgb.Dataset(X_train, label=y_train, group=train_list)
        lgb_eval = lgb.Dataset(X_eval, label=y_eval, reference=lgb_train, group=eval_list)

        # '''
        # クロスバリデーションによるハイパーパラメータの探索 3fold
        tuner = lgb.LightGBMTunerCV(params,
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
            'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
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
        model = lgbm.train(params,
                        lgb_train,  # トレーニングデータの指定
                        valid_names=['train', 'valid'],     # 学習経過で表示する名称
                        valid_sets=[lgb_train, lgb_eval],
                        # categorical_feature = cat_list,
                        callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),
                                    lgbm.record_evaluation(evals_result)]
                        )

        # pklファイルとしてモデルを保存
        with open(f"{tuner_path}{seed}.pickle", "wb") as mk:
            pickle.dump(model, mk)

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(X_test, group=test_list)
        y_test_id[f'result{seed}'] = y_pred
    print(len(test_list))

    return y_test_id

# スタッキング
def stacking(y_test_id):
    srate = 0.01

    # 説明変数追加
    temp = 0
    Z = y_test_id.iloc[:, 6:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['レースID'] = y_test_id['レースID']
    Z['rank'] = y_test_id['rank']
    Z = Z.sort_values(['レースID', 'Average'], ascending=[True, False])

    # クエリListを作成
    id_count = Z['レースID'].value_counts(sort=False)
    train2_list = id_count.values.tolist()

    n_list = train2_list[:-1]
    n = 0
    for i in n_list:
        n += i
        for k in range(1, 6):
            mean_df = Z.iloc[n-i:n, Z.columns.get_loc(f'result{k}')].mean()
            std_df = Z.iloc[n-i:n, Z.columns.get_loc(f'result{k}')].std()
            Z.iloc[n-i:n, Z.columns.get_loc(f'result{k}')] = (Z.iloc[n-i:n, Z.columns.get_loc(f'result{k}')] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['レースID', 'Average'], ascending=[True, False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    for i in train2_list:
        Z.iat[i+temp-1, -2] = float('nan')
        Z.iat[temp, -1] = float('nan')
        temp += i

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    Z = Z.sort_values(['レースID'], ascending=True)

    # # トレーニングデータ,テストデータの分割202102011001
    # Z_train, Z_test = Z[Z['レースID'] < 202102011000], Z[Z['レースID'] >= 202102011000] # 函館
    # Z_train, Z_test = Z[Z['レースID'] < 202105040000], Z[Z['レースID'] >= 202105040000] # 東京
    # Z_train, Z_test = Z[Z['レースID'] < 202107040000], Z[Z['レースID'] >= 202107040000] # 中京
    Z_train, Z_test = Z.iloc[:int(len(Z) / 5), :], Z.iloc[int(len(Z) / 5):, :]
    w_train, w_test = Z_train['rank'], Z_test['rank']

    # クエリListを作成
    id_count = Z_train['レースID'].value_counts(sort=False)
    train2_list = id_count.values.tolist()

    id_count = Z_test['レースID'].value_counts(sort=False)
    test2_list = id_count.values.tolist()

    # レースIDカラムを削除
    Z_train = Z_train.drop(['レースID', 'rank'], axis=1)
    Z_test = Z_test.drop(['レースID', 'rank'], axis=1)

    # print(Z_train.shape)
    # print(Z_test.shape)

    # カラム名削除
    # Z_train = Z_train.values
    # Z_test = Z_test.values
    # w_train = w_train.values
    # w_test = w_test.values

    # 学習に使用するデータを設定
    zlgb_train = lgb.Dataset(Z_train, label=w_train, group=train2_list)
    zlgb_test = lgb.Dataset(Z_test, label=w_test, reference=zlgb_train, group=test2_list)

    for seed in range(1, file_num+1):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',  # ←ここでランキング学習と指定！
            'metric': 'ndcg',   # for lambdarank
            'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
            'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
            'label_gain': gain_list,
            'learning_rate': srate,
            'random_state': seed,
            'verbose_eval': 20,
            'early_stopping_round': 20,
            'n_estimators': 10000,
        }

        # データセット再構築
        del zlgb_train
        del zlgb_test
        gc.collect()
        zlgb_train = lgb.Dataset(Z_train, label=w_train, group=train2_list)
        zlgb_test = lgb.Dataset(Z_test, label=w_test, reference=zlgb_train, group=test2_list)
        
        tuner = lgb.LightGBMTunerCV(params,
                                    zlgb_train,
                                    folds=GroupKFold(n_splits=3),
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
            'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
            'label_gain': gain_list,
            'learning_rate': srate,
            'random_state': seed,
            'verbose_eval': 20,
            'early_stopping_round': 10,
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
        model_s = lgbm.train(params,
                        zlgb_train,  # トレーニングデータの指定
                        valid_names=['train', 'valid'],     # 学習経過で表示する名称
                        valid_sets=[zlgb_train, zlgb_test],
                        callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),
                                    lgbm.record_evaluation(evals_result)]
                        )
        
        # pklファイルとしてモデルを保存
        # if seed == 1 or seed == 8:
        with open(f"{tuner_path}{seed}-stack.pickle", "wb") as mk:
            pickle.dump(model_s, mk)



# データフレーム生成
df = pd.DataFrame()

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)

# 列を全表示（列の数）
pd.set_option("display.max_columns", None)

# csvファイル読み込み(スクレイピングしない場合)
df = pd.read_csv(csv_path, index_col=0)
print(df.columns)

# 辞書作成(各コースの平均タイム)
speed_dict = {'112001': 68.5, '116001': 94.3, '118001': 108.5, '120001': 121.2, '122001': 133.8, '125001': 153.5, '136001': 227.0,
              '112002': 71.0, '118002': 113.8, '124002': 154.2, '125002': 162.5, '214001': 81.0, '216001': 93.9, '218001': 106.9, '220001': 120.3,
              '224001': 145.8, '225001': 151.1, '234001': 213.6, '213002': 77.2, '214002': 83.5, '216002': 97.0, '221002': 130.5, '312001': 68.3,
              '314001': 81.4, '316001': 94.0, '318001': 107.4, '320001': 120.4, '322001': 133.2, '324001': 145.9, '330001': 187.0,
              '332001': 195.1, '312002': 71.4, '314002': 83.8, '318002': 111.3, '319002': 117.2, '412001': 68.3, '414001': 81.0,
              '416001': 94.2, '418001': 106.5, '420001': 120.6, '422001': 133.2, '424001': 156.7, '426001': 159.5, '430001': 185.2, '412002': 70.7,
              '414002': 83.6, '418002': 111.4, '420002': 124.5, '512001': 69.0, '515001': 89.5, '518001': 107.5, '520001': 121.6,
              '526001': 161.6, '510002': 57.55, '517002': 103.8, '524002': 155.7, '610001': 57.7, '612001': 68.6, '618001': 107.6, '620001': 120.9, '626001': 161.6,
              '610002': 58.20, '617002': 104.8, '624002': 154.7, '712001': 68.8, '718001': 108.0, '720001': 120.4, '726001': 141.3, '711502': 67.6,
              '717002': 104.5, '724002': 155.2, '810001': 54.9, '812001': 69.0, '814001': 80.8, '816001': 93.7, '818001': 107.1, '820001': 120.0,
              '822001': 134.1, '824001': 146.9, '812002': 70.7, '818002': 112.2, '825002': 161.4, '912001': 68.2, '914001': 81.1, '916001': 93.9,
              '920001': 120.9, '922001': 134.7, '912002': 71.2, '914002': 83.8, '918002': 111.0, '919002': 119.4, '1012001': 69.0,
              '1015001': 89.5, '1017001': 161.1, '1018001': 107.5, '1020001': 121.6, '1026001': 161.6, '1010002': 57.55, '1017002': 103.8, '1024002': 153.8}

# df = df[df['レースID'] < 202200000000]

df = df.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し

horse_mapping = create_unique_pickle(df['騎手'], jockey_path)