import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.discriminant_analysis import StandardScaler
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
import random
import seaborn as sns

import Listwise

def save_csv(path, df_all):
    df_all.to_csv(path, na_rep='NaN')

# ターゲットエンコーディング
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

# floatに変換
def convert_to_float_if_possible(df):
    df_converted = df.copy()
    for col in df.columns:
        try:
            # 変換を試みる（NaNは発生させたくないので errors='raise'）
            converted = pd.to_numeric(df[col], errors='raise')
            # print(col, converted.dtype)
            if pd.api.types.is_numeric_dtype(converted):
                df_converted[col] = converted.astype(float)
        except:
            pass  # 変換できなかった列は無視
    return df_converted

# スクレイピング
def scraping(csv_path, no, start, end):
    df = pd.DataFrame()
    # ヘッダー
    headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    }

    session = requests.Session()
    session.headers.update(headers)

    for year in range(start, end):
        for number in range(1, 6):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    race_id = '{}{}{}{}{}'.format(str(year), no, str(number).zfill(2), str(day).zfill(2), str(race_no).zfill(2))
                    url_race = 'https://race.netkeiba.com/race/result.html?race_id={}&rf=race_list'.format(race_id)
                    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
                    try:
                        response_race = session.get(url_race, headers=headers)
                        response_past = session.get(url_past, headers=headers)
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

def scraping_local(csv_path, no, start, end):
    df = pd.DataFrame()
    # ヘッダー
    headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    }

    session = requests.Session()
    session.headers.update(headers)
    for year in range(start, end):
        for month in range(1, 13):
            for day in range(1, 32):
                for race_no in range(1, 13):
                    race_id = '{}{}{}{}{}'.format(str(year), no, str(month).zfill(2), str(day).zfill(2), str(race_no).zfill(2))
                    # race_id = "201530042201"
                    url_race = 'https://nar.netkeiba.com/race/result.html?race_id={}&rf=race_list'.format(race_id)
                    url_past = 'https://nar.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
                    try:
                        response_race = session.get(url_race, headers=headers)
                        response_past = session.get(url_past, headers=headers)
                        df_result = pd.read_html(response_race.content)[0]
                        df_past = pd.read_html(response_past.content)[0]
                        soup = BeautifulSoup(response_race.content, 'html.parser')
                        data1 = soup.find('div', class_='RaceData01').text
                        data2 = soup.find('div', class_='RaceData02').text
                        data3 = soup.find('tr', class_='Umatan').text
                        data4 = soup.find('div', class_='RaceName').text
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
                        if race_no == 1:
                            print("no:"+url_race)
                            break
                        continue
                    df_result_past['レースID'] = race_id
                    df = pd.concat([df, df_result_past])

    # 結果をcsvに保存
    df.to_csv(csv_path, na_rep='NaN')

def scrape_payouts_combination(csv_path, no):
    df = pd.DataFrame(columns=['レースID', '券種', '馬番', '払い戻し金額'])

    headers = {'User-Agent': 'Mozilla/5.0'}

    for year in range(2012, 2025):
        for number in range(1, 6):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    race_id = '{}{}{}{}{}'.format(str(year), no, str(number).zfill(2), str(day).zfill(2), str(race_no).zfill(2))
                    url_race = f'https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_list'
                    try:
                        response = requests.get(url_race, headers=headers)
                        soup = BeautifulSoup(response.content, 'html.parser')

                        data2 = soup.find('div', class_='RaceData02').text
                        a = data2[data2.find('新馬')+0: data2.find('新馬')+2]
                        if a == '新馬':
                            continue

                        payout_tables = soup.find_all('table', class_='Payout_Detail_Table')
                        
                        for table in payout_tables:
                            for tr in table.find_all('tr'):
                                bet_type = tr.find('th').text.strip()  # 券種
                                payout_td = tr.find('td', class_='Payout')
                                result_td = tr.find('td', class_='Result')

                                if not payout_td or not result_td:
                                    continue

                                # 的中馬番
                                combos = []
                                if result_td.find_all('ul'):
                                    # 複数馬番の組み合わせが <ul><li><span> の形式で複数ある場合
                                    for ul in result_td.find_all('ul'):
                                        horses = [span.text.strip() for span in ul.find_all('span') if span.text.strip()]
                                        if horses:
                                            combos.append('-'.join(horses))
                                else:
                                    # 単勝・複勝など <div><span> 形式
                                    horses = [span.text.strip() for span in result_td.find_all('span') if span.text.strip()]
                                    for h in horses:
                                        combos.append(h)

                                # 払い戻し金額
                                payout_text = payout_td.get_text(separator='|')  # 改行やbrを | で区切る
                                payouts = [p.strip().replace('円','').replace(',','') for p in payout_text.split('|') if p.strip()]

                                # コンボと払い戻し金額を対応
                                for i, combo in enumerate(combos):
                                    if i < len(payouts):
                                        payout_amount = payouts[i]
                                    else:
                                        payout_amount = ''
                                    df = pd.concat([df, pd.DataFrame([{
                                        'レースID': race_id,
                                        '券種': bet_type,
                                        '馬番': combo,
                                        '払い戻し金額': payout_amount
                                    }])], ignore_index=True)

                        print(f"Scraped {race_id}")
                        # print(df)

                        time.sleep(1)
                    except Exception as e:
                        print(f"Error {race_id}: {e}")
                        continue

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved to {csv_path}")

# 辞書作成
def create_unique_pickle(series, file_path):
    mapping = dict(zip(series.unique().tolist(), range(1, len(series.unique().tolist()) + 1)))
    with open(file_path, "wb") as jd:
        pickle.dump(mapping, jd)
    
    return mapping

# 辞書読み込み
def return_pickle(file_path):
    with open(file_path, "rb") as jd:
        mapping = pickle.load(jd)
    
    return mapping

# データの初期加工
def df_first_processing(df, name, type='推論'):
    horse_path = f"./pickle-dict/horse_jra_{name}.pkl" # 父馬のマッピング用辞書のパス
    jockey_path = f"./pickle-dict/jockey_jra_{name}.pkl" # 騎手のマッピング用辞書のパス

    df = df.copy()
    # 新たなカラムを作成
    if df['場所'].iloc[0] <= 10:
        df['父馬'] = df['馬名_y'].str.extract(r'(\w+\s)', expand=True)
        df['間隔'] = df['馬名_y'].str.extract(r'中\s*(\d+)\s*週', expand=True).astype(float)
    else:
        df['父馬'] = df['馬名 オッズ'].str.extract(r'(\w+\s)', expand=True)
        df['間隔'] = df['馬名 オッズ'].str.extract(r'中\s*(\d+)\s*週', expand=True).astype(float)
    # print(df['間隔'])

    # 血統pickle作成
    if type != '推論':
        horse_mapping = create_unique_pickle(df['父馬'], horse_path)
        # マッピング
        df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    else:
        horse_mapping = return_pickle(horse_path)
        # マッピング
        df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')

    # 馬単の配当を抽出
    if '馬単' in df.columns:
        df['馬単'] = df['馬単'].str.replace(',', '')
        df['馬単'] = df['馬単'].str.extract(r'(\d+円)', expand=True)
        df['馬単'] = df['馬単'].str.extract(r'(\d+)', expand=True)

    # いらないカラムを消す
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', 'コーナー通過順', '厩舎', 'タイム', '騎手斤量', '着差', '後3F', '印', '馬名 オッズ', '馬名'], axis=1, errors="ignore")

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
    if type != '推論':
        jockey_mapping = create_unique_pickle(df['騎手'], jockey_path)
        df['騎手'] = df['騎手'].map(jockey_mapping)

        # df['勝率'] = 0
        # df['勝率'] = df['勝率'].mask(pd.to_numeric(df['着順'], errors='coerce') == 1, 1)

        # d = df.groupby('騎手')['勝率'].mean()
        # dict = d.to_dict()
        # df['騎手'] = pd.to_numeric(df['騎手'].map(dict), errors='coerce')

        # with open(f'./pickle-dict/jwin_dict_{name}.pkl', "wb") as dd:
        #     pickle.dump(dict, dd)
    else:
        jockey_mapping = return_pickle(jockey_path)
        df['騎手'] = df['騎手'].map(jockey_mapping)

        # j_win = return_pickle(f'./pickle-dict/jwin_dict_{name}.pkl')
        # df['騎手'] = df['騎手'].map(j_win)

    

    # 「性齢」「馬体重（増減）」はいらないので消す
    df = df.drop(['性齢', '馬体重(増減)', '馬体重', '体重増減'], axis=1)

    df = convert_to_float_if_possible(df)

    return df

# 前走~5走前のデータを処理
def df_big_past_processing(df, name, field_num):
    jockey_path = f"./pickle-dict/jockey_jra_{name}.pkl" # 騎手のマッピング用辞書のパス

    df_all = df.copy()
    # 騎手の辞書を読み込み
    jockey_mapping = return_pickle(jockey_path)

    # 以降1~5走の処理
    for sou in range(1, 6):
        sou = str(sou)
        if int(sou) == 1:
            df_split = df['前走'].astype('object').str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
        else:
            df_split = df[sou+'走'].astype('object').str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
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
        for i in range(1, 26):
            for k in range(800, 3700, 10):
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

        # print(df_split[sou+'場所'].dtype, df_split[sou+'距離'].dtype, df_split[sou+'フィールド'].dtype, df['フィールド'].dtype, df['距離'].dtype)
        df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
        df_split[sou+'場所変化'] = df_split[sou+'場所'] - field_num
        df_split[sou+'フィールド変化'] = df_split[sou+'フィールド'] - df['フィールド']

        # 不要なカラムを削除
        # df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)
        df_split = df_split.drop([sou+'レース名', sou+'騎手'], axis=1)


        # 今走と過去走を結合
        df_all = pd.concat([df_all, df_split], axis=1)

        if int(sou) == 1:
            past_level(df_all)
        
    # print(df_all.head(10))

    return df_all

# 過去走の平均クラスと平均ペースを算出
def past_level(df_all, type='推論'):
    df_all = df_all.copy()
    if type != '推論':
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
    else:
        df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
        df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

        df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
        df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']

    return df_all

# テストデータを分離してターゲットエンコーディング
def encording(df_all):
    df_all = df_all.copy()
    df_test = df_all[df_all['レースID'] >= 202000000000].copy()
    df_all = df_all[df_all['レースID'] < 202000000000]

    d = df_all.groupby('父馬')['rank'].mean()
    dict = d.to_dict()
    df_test['父馬'] = pd.to_numeric(df_test['父馬'].astype(float).map(dict), errors='coerce')
    with open(f'./pickle-dict/sire_dict{field}.pkl', "wb") as dd:
        pickle.dump(dict, dd)
    
    df_all = target_encording(df_all, '父馬', 'rank')

    df_all = pd.concat([df_all, df_test], axis=0)

    return df_all

# 終盤のデータ加工
def df_end_processing(df_all, type='推論'):
    df_all = df_all.copy()
    if type != '推論':
        # 着順から文字列を排除
        indexNames = df_all[(df_all['着順'] != '中止') & (df_all['着順'] != '除外') & (df_all['着順'] != '取消') & (df_all['着順'] != '失格') & (df_all['着順'] != '未定')]
        df_all = indexNames

    # 着差とスピード指数のbest, avカラム作成
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    df_all = df_all.drop(['前走', '2走', '3走', '4走', '5走', 'レース名', '勝率'], axis=1, errors='ignore')

    # print(df_all.isnull().sum())

    # # 空白削除
    # for i in df_all.columns:
    #     df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    cols = ['5過去着順', '4過去着順', '3過去着順', '2過去着順', '1過去着順']

    df_all[cols] = df_all[cols].apply(pd.to_numeric, errors='coerce')
    df_all['上昇度'] = (df_all['5過去着順'] - df_all['4過去着順']) + (df_all['4過去着順'] - df_all['3過去着順']) + (df_all['3過去着順'] - df_all['2過去着順']) + (df_all['2過去着順'] - df_all['1過去着順']) / (df_all[['1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順']].isnull().sum(axis=1) + 1)
    
    # df_all['rank'] = df_all['着順'].map(f_ranking)
    df_all = df_all.replace(['', '未定', '除外', '取消', '失格', '中止'], np.nan)
    # df_all = df_all.replace('', '00000')  # ''をエラー検出文字に置換してくれる
    # df_all = df_all.replace('未定', '00000')
    # df_all = df_all[~df_all.astype(str).apply(lambda s: s.str.contains('00000', na=False), axis=1).any(axis=1)]  # エラー検出文字を入れた行以外を抽出
    # df_all = df_all.astype(float)

    if type != '推論':
        # 1着のみ単勝オッズを保有
        df_all['オッズ'] = df_all['単勝オッズ']
        df_all['単勝オッズ'] = df_all['単勝オッズ'].where(df_all['着順'].astype(int) == 1, 0)
        df_all['単勝オッズ'] = df_all['単勝オッズ'] * 100

        df_all['rank'] = df_all['単勝オッズ']

        # レース内の順序をシャッフル
        df_all = df_all.sample(frac=1)
        df_all = df_all.sort_values(['レースID'], ascending=True)

    # 型変換
    df_all = convert_to_float_if_possible(df_all)
    
    return df_all

# label_gain作成
def create_label_gain(df_all):
    n_bins = 18
    df_all = df_all.copy()
    # 例）着順があるdfに対して、"gain"を相対スコアで計算
    df_all['rank'] = df_all['着順'].astype(float).apply(lambda r: 1 / r)  # 単純逆数
    # 出走頭数で正規化してもよい
    df_all['rank'] *= df_all['出走頭数'].astype(float) / n_bins
    # ランクをラベル化
    bins = np.linspace(0, 1, df_all['rank'].nunique())  # ランクの境界値を計算
    df_all['rank'] = np.digitize(df_all['rank'], bins, right=True) - 1
    # gainを計算
    label_gain = [np.sqrt(x) for x in range(0, df_all['rank'].max() + 1)]
    
    return df_all, label_gain

# def create_label_gain(df_all, top_k=5):
#     """
#     各レース内で着順に応じたrelevanceラベルとlabel_gainを生成
#     """
#     df_all = df_all.copy()
    
#     # --- レースごとに relevance を作成（上位ほど高い値）---
#     df_all['rank'] = df_all.groupby('レースID')['着順'] \
#         .transform(lambda x: len(x) - x.rank(method='first') + 1)
    
#     # --- 0始まり整数に変換（LightGBM要件）---
#     df_all['rank'] = df_all['rank'].astype(float) - 1

#     # --- gain設定：上位を強調 ---
#     max_rel = int(df_all['rank'].max())  # ← 型キャストを追加！
#     label_gain = [int(2 ** x - 1) for x in range(max_rel + 1)]

#     return df_all, label_gain

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
    # X_train = df_all.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    # X_test = df_test.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    # y_train = df_all[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]
    # y_test = df_test[['rank', 'レースID', 'オッズ', '着順', '単勝オッズ', '馬単']]

    # # train, eval, testに分割
    # X_eval = df_eval.drop(['着順', 'rank', 'オッズ', '単勝オッズ', '馬単'], axis=1)
    # y_eval = df_eval[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]

    # # クエリListを作成
    id_count = df_all['レースID'].value_counts(sort=False)
    train_list = id_count.values.tolist()

    id_count = df_test['レースID'].value_counts(sort=False)
    test_list = id_count.values.tolist()

    id_count = df_eval['レースID'].value_counts(sort=False)
    eval_list = id_count.values.tolist()

    # print("train_list 前半:", train_list[:10])
    # print("train_list の合計:", sum(train_list))
    # print("X_train の長さ:", len(df_all))
    # print("一致チェック:", sum(train_list) == len(df_all))

    # # 検証用のレースIDを保存
    # y_test_id = pd.DataFrame()
    # y_test_id['レースID'] = y_test['レースID']
    # y_test_id['着順'] = y_test['着順']
    # y_test_id['単勝オッズ'] = y_test['単勝オッズ']
    # y_test_id['オッズ'] = y_test['オッズ']
    # y_test_id['馬単'] = y_test['馬単'].astype(int)
    # y_test_id['rank'] = y_test['rank']

    # # レースIDカラムを削除
    # X_train = X_train.drop(['レースID'], axis=1)
    # X_test = X_test.drop(['レースID'], axis=1)
    # X_eval = X_eval.drop(['レースID'], axis=1)
    # y_train = y_train.drop(['レースID', '着順', '単勝オッズ', '馬単'], axis=1)
    # y_test = y_test.drop(['レースID', '着順', '単勝オッズ', '馬単', 'オッズ'], axis=1)
    # y_eval = y_eval.drop(['レースID', '着順', '単勝オッズ', '馬単'], axis=1)

    # # dataframeを値のみに
    # display(list(X_train.columns.values))

    # パラメータ設定
    rate = 0.01
    lgb_train = lgb.Dataset(df_all[feature_cols], label=df_all['rank'], group=train_list)
    lgb_eval = lgb.Dataset(df_eval[feature_cols], label=df_eval['rank'], reference=lgb_train, group=eval_list)

    for seed in range(1, file_num+1):
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
        }
        ####################################################################################

        # データセット再構築
        del lgb_train
        del lgb_eval
        gc.collect()
        lgb_train = lgb.Dataset(df_all[feature_cols], label=df_all['rank'], group=train_list)
        lgb_eval = lgb.Dataset(df_eval[feature_cols], label=df_eval['rank'], reference=lgb_train, group=eval_list)

        # '''
        # クロスバリデーションによるハイパーパラメータの探索 3fold
        tuner = lgb.LightGBMTunerCV(params,
                                    lgb_train,
                                    folds=GroupKFold(n_splits=3),
                                    categorical_feature = cat_list,
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
        model = lgbm.train(params,
                        lgb_train,  # トレーニングデータの指定
                        valid_names=['train', 'valid'],     # 学習経過で表示する名称
                        valid_sets=[lgb_train, lgb_eval],
                        categorical_feature = cat_list,
                        callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),
                                    lgbm.record_evaluation(evals_result)]
                        )

        # pklファイルとしてモデルを保存
        # with open(f"{tuner_path}{seed}.pickle", "wb") as mk:
        #     pickle.dump(model, mk)

        # 学習後のモデル
        importance_gain = model.feature_importance(importance_type="gain")  # 各特徴量の寄与度
        importance_split = model.feature_importance(importance_type="split")  # 分割に使われた回数

        feature_names = df_all[feature_cols].columns

        feat_imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance_gain": importance_gain,
            "importance_split": importance_split
        }).sort_values(by="importance_gain", ascending=False)

        print(feat_imp_df)  # 上位20特徴量

        plt.figure(figsize=(10,6))
        sns.barplot(x="importance_gain", y="feature", data=feat_imp_df)
        plt.title("Top 20 Feature Importance (LambdaRank)")
        plt.show()

        # テストデータの予測 (予測クラスを返す)
        y_pred = model.predict(df_test[feature_cols], group=test_list)
        y_test_id[f'result{seed}'] = y_pred
    print(len(test_list))

    return y_test_id, test_list

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
            'ndcg_eval_at': [1],  # 3連単を予測したい
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
            'ndcg_eval_at': [1],  # 3連単を予測したい
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
        # with open(f"{tuner_path}{seed}-stack.pickle", "wb") as mk:
        #     pickle.dump(model_s, mk)

def sort(result, num):
    result = result.copy()
    result = result.sort_values(['レースID', f'score{num}'], ascending=[True, False])
    result = result.reset_index(drop=True)

    return result

def test(test_list, result, file_num):
    
    for i in range(1, file_num+1):
        # result = sort(result, i)
        result_list = []
        sum = 0
        hit = 0
        count = 0
        cursor = 0

        for k in test_list:
            # スコア抽出
            subset = result.iloc[cursor:cursor+k, result.columns.get_loc(f'result{i}')].values

            # softmax 計算（安定化のため最大値を引く）
            exp_scores = np.exp(subset - np.max(subset))
            softmax_values = exp_scores / exp_scores.sum()

            # 結果を DataFrame に格納（例として新しい列に）
            result[f'softmax{i}'] = np.nan
            result.iloc[cursor:cursor+k, result.columns.get_loc(f'softmax{i}')] = softmax_values 

            cursor += k

        # 期待値
        result[f'score{i}'] = result['オッズ'] * result[f'softmax{i}']
        cursor = 0
        result = sort(result, i)
        for v in test_list:
            sum += result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float)
            if result.iloc[cursor, result.columns.get_loc('単勝オッズ')].astype(float) > 0:
                hit += 1
            count += 1
            cursor += v
        
        result_list.append(sum / (count * 100))
        print(f'回収率:{sum / (count * 100)}')
        print(f'的中率:{hit / count}')
    
    mean_return = np.mean(result_list)
    std_return = np.std(result_list)

    print(f"平均回収率: {mean_return:.2%} ± {std_return:.2%}")
    


nagoya_mapping = {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
                        '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
                        '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

class_dict = {'GI': 1, 'GII': 2, 'GIII': 3, 'OP': 4, 'L': 4, '3勝': 5, '1600万下': 5, '1600下': 5, '2勝': 6, '1000万下': 6, '1000下': 6,
                '1勝': 7, '500万下': 7, '500下': 7, '未勝利': 8, '新馬': 9}

field_mapping = {'芝': 1, 'ダ': 2, '障': 3}

condition_mapping = {'良': 1, '稍': 2, '重': 3, '不': 4}

base_time = pd.to_datetime('00:00.0', format='%M:%S.%f')


'''
JRAソース
https://datsusara-horse.com/2021/08/25/keiba_course_time/

地方ソース
https://kaisekisya.net/local/index.html
https://www.keiba.go.jp/guide/course_record/

基本，条件戦で一番グレードの高いレースを参考にしている
'''
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
                '1015001': 89.5, '1017001': 161.1, '1018001': 107.5, '1020001': 121.6, '1026001': 161.6, '1010002': 57.55, '1017002': 103.8, '1024002': 153.8,
                '1210002': 61.0, '1211002': 68.0, '1212002': 74.1, '1215002': 99.1, '1216002': 102.5, '1217002': 110.4, '1218002': 118.4, '1220002': 133.8, '1226002': 174.9,
                '1310001': 58.2, '1316001': 97.6, '1317001': 105.5, '1324001': 149.6, '1310002': 60.1, '1312002': 73.5, '1314002': 86.3, '1316002': 99.4, '1318002': 113.6, '1320002': 123.9, '1326002': 170.0,
                '148502': 51.5, '1413002': 83.7, '1414002': 89.5, '1416002': 102.6, '1418002': 117.5, '1419002': 123.6, '1420002': 127.8,
                '158002': 46.8, '1513002': 82.9, '1514002': 88.0, '1515002': 95.0, '1516002': 104.0, '1520002': 131.1,
                '1610002': 60.3, '1612002': 74.0, '1615002': 96.5, '1616002': 102.7, '1617002': 110.0, '1618002': 116.1, '1622002': 146.0, '1624002': 158.1,            
                '1710002': 60.1, '1712002': 72.8, '1714002': 86.5, '1716002': 101.1, '1716502': 106.7, '1717002': 108.6, '1718002': 114.3, '1720002': 127.4, '1724002': 156.5, '1726002': 172.8,
                '189002': 54.0, '1814002': 90.6, '1815002': 95.4, '1816002': 103.3, '1820002': 130.9, '1821002': 142.3,
                '199002': 55.7, '1914002': 88.2, '1915002': 94.8, '1917002': 109.6, '1919002': 124.0, '1920002': 128.6, '1921002': 135.3, '1926002': 171.0,
                '208002': 50.7, '2014002': 89.3, '2016002': 103.7, '2018002': 117.7, '2019002': 126.6, '2025002': 170.6,
                '219002': 55.3, '219202': 55.3, '2114002': 89.1, '2115002': 95.9, '2117002': 110.3, '2120002': 132.2, '2121002': 138.7,
                '228202': 49.7, '2212302': 79.2, '2214002': 90.2, '2217002': 113.3, '2218702': 125.4, '2224002': 169.4,
                '238002': 49.7, '2314002': 92.2, '2315002': 97.6, '2318002': 123.6, '2320002': 137.1,
                '248002': 49.0, '2413002': 85.1, '2414002': 91.6, '2416002': 106.7, '2418002': 121.0, '2419002': 130.8, '2424002': 169.9,
                '259002': 53.1, '2513002': 82.7, '2514002': 89.5, '2517502': 115.0, '2518002': 119.1, '2518602': 123.0, '2520002': 132.4, '2525002': 171.2}
                

if __name__ == "__main__":
    warnings.simplefilter('ignore')

    optuna.logging.disable_default_handler()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)

    # 開催場所番号
    field = 1
    field_name = 'nakayama000'
    csv_path = f"./csv/{field_name}_2012-2024.csv" # 学習に使うcsvデータのパス
    file_num = 1

    # {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
    #  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
    #  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

    # データフレーム生成
    df = pd.DataFrame()

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)
    # セルの文字列を省略せずに全部表示
    pd.set_option("display.max_colwidth", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # csvファイル読み込み(スクレイピングしない場合)
    # csv_files = [
    #     './csv/nakayama_2012-2024.csv',
    #     './csv/tokyo_2012-2024.csv',
    #     './csv/kyoto_2012-2024.csv',
    #     './csv/hanshin_2012-2024.csv',
    #     './csv/sapporo_2012-2024.csv',
    #     './csv/hakodate_2012-2024.csv',
    #     './csv/hukushima_2012-2024.csv',
    #     './csv/nigata_2012-2024.csv',
    #     './csv/chukyo_2012-2024.csv',
    #     './csv/kokura_2012-2024.csv'
    # ]

    # dfs = []
    # for i, path in enumerate(csv_files, start=1):
    #     df_fold = pd.read_csv(path, index_col=0)
    #     df_fold['場所'] = i
    #     dfs.append(df_fold)
    #     print(f"レース数: {len(df_fold['レースID'].unique())}")

    # df = pd.concat(dfs, ignore_index=True)

    # # 京都改修前削除
    # df = df.drop(df[(df["場所"] == 3) & (df["レースID"].astype(int) < 202308010101)].index)


    df = pd.read_csv(csv_path, index_col=0)
    df1 = pd.read_csv(f"./csv/nakayama_2025.csv", index_col=0)

    df = pd.concat([df, df1], axis=0).reset_index(drop=True)
    # print(pd.Series(sorted(df['レースID'].unique(), reverse=True)[:5]))

    df['場所'] = field
    # print(df.columns)
    # print(df.head(5))

    # 辞書作成(各コースの平均タイム)

    # df = df[df['レースID'] < 202200000000]

    df = df.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し

    #############################################ここから処理開始###############################################################
    
    # 今走の処理
    df = df_first_processing(df, field_name, type='a')
    print(df['父馬'].head(10))
    # df = df_first_processing(df, field_name)
    # 過去走の処理
    df_all = df_big_past_processing(df, field_name, field)
    # 過去のレベル
    df_all = past_level(df_all, type='a')
    # df_all = past_level(df_all)
    # 終了処理
    df_all = df_end_processing(df_all, type='b')
    # df_all = df_end_processing(df_all)
    # csv
    save_csv(f'./csv/df_all_{field_name}_2025.csv', df_all)
    # ラベリング
    # df_all = encording(df_all)
    # ラベル分割
    df_all, gain_list = create_label_gain(df_all)

    df = Listwise.inversion(df_all)
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    # feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    feature_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
    
    pkl_path = f'./pickle-dict/sire_dict_{field_name}_fold0.pkl'
    with open(pkl_path, "rb") as f:
        sire_mapping = pickle.load(f)
    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])
    df = Listwise.fill_nan(df, feature_cols)

    # ----------------------------
    # 場所列追加
    # ----------------------------
    # df['場所'] = field_num

    # ----------------------------
    # カテゴリ列数値化
    # ----------------------------
    df = Listwise.race_feature(df)

    cat_list = Listwise.embedding_cols + Listwise.context_cat_cols

    # feature_cols = [col for col in feature_cols if col not in Listwise.embedding_cols and col not in Listwise.common_cols]

    # 学習
    y_test_id, test_list = first_train(df)
    # test
    test(test_list, y_test_id, file_num)

    # スタッキング
    # stacking(y_test_id)