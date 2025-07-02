import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pickle
from IPython.display import display
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from decimal import Decimal
import category_encoders as ce

# 193 = 1.96^2 * (3.57 - 1) / r^2
# 1088 = 1.96^2 * (3.45 - 1) / r^2
warnings.simplefilter('ignore')

# カラム順
# ['馬番' '斤量' '騎手' '距離' 'フィールド' '馬場' '出走頭数' 'レースID' '父馬' '間隔' '性' '齢' '体重増減'
#  '1場所' '1過去着順' '1フィールド' '1距離' '1馬場' '1後3F' '1着差' '1クラス' '1人気差' '2過去着順'       
#  '2後3F' '2着差' '2クラス' '2人気差' '3過去着順' '3後3F' '3着差' '3クラス' '3人気差' '4過去着順'
#  '4後3F' '4着差' '4クラス' '4人気差' '5過去着順' '5後3F' '5着差' '5クラス' '5人気差']

# {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
#  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
#  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

# 1.7: 差が0.4未満(0.3-0.4 good) 103%
# 1.1-1.6: 0.6未満(0.2-0.3 good) 115-125%
# 1: 差が1未満(0.4-0.5 good) 157%

# 会場
# field = 'hakodate'
# field_num = 6

# field = 'tyukyo'
# field_num = 9

# field = 'tokyo'
# field = 'nakayama'
field_num = 4

# tuner_name = "tokyo"
# field2 = 'tyukyo'

tuner_name = "hanshin"
file_name = 'hanshin'
# file_name2 = 'tyukyo'
# file_name3 = 'tokyo'
# file_name4 = 'hanshin'
# file_name5 = 'nakayama'

# ファイル数
file_num = 5

# ファイルパス
csv_path = f"./csv/{file_name}_2012-2024.csv" # 学習に使うcsvデータのパス
# csv_path2 = f"./csv/{file_name2}_2012-2024.csv" # 学習に使うcsvデータのパス
# csv_path3 = f"./csv/{file_name3}_2012-2024.csv" # 学習に使うcsvデータのパス
# csv_path4 = f"./csv/{file_name4}_2012-2024.csv" # 学習に使うcsvデータのパス
# csv_path5 = f"./csv/{file_name5}_2012-2024.csv" # 学習に使うcsvデータのパス

version = 'test70'
horse_path = "./pickle-dict/horse_jra.pkl"
femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
jockey_path = "./pickle-dict/jockey_jra.pkl"
tuner_path = f"./pickle-tuner/{tuner_name}{version}_"
# csv_path = f"./csv/{field}_2012-2024.csv"
# csv2_path = f"./csv/{field2}_2012-2024.csv"
fname = "./pickle-dict/corpus.pkl"

# 行を全表示（行の数）
# pd.set_option("display.max_rows", None)
pd.options.display.max_rows = None

# 列を全表示（列の数）
# pd.set_option("display.max_columns", None)
pd.options.display.max_columns = None

# list全表示
np.set_printoptions(threshold=np.inf)

# 関数
def Target_encording(df, column, target):
    tem = pd.DataFrame()
    dfs = [df.loc[i:i+k-1, :] for i in range(0, len(df.index), int(len(df.index) / 5) + 1)]

    for i in dfs:
        d = i.groupby(column)[target].mean()
        dict = d.to_dict()
        i[column] = pd.to_numeric(i[column].map(dict), errors='coerce')
        tem = pd.concat([tem, i], axis=0)
    
    return tem

# データフレームを生成
df_shutuba = pd.DataFrame()
df2_shutuba = pd.DataFrame()

df_shutuba = pd.read_csv(csv_path, index_col=0)
# df_shutuba['場所'] = 3
# name_list = [csv_path2, csv_path3, csv_path4, csv_path5]

# for i in name_list:
#     df2 = pd.read_csv(i, index_col=0)
#     df_shutuba = pd.concat([df_shutuba, df2])
# df2_shutuba = pd.read_csv(csv2_path, index_col=0)
# df_shutuba = pd.concat([df_shutuba, df2_shutuba])
df_shutuba = df_shutuba.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し

df_shutuba = df_shutuba[df_shutuba['レースID'] >= 202200000000]
# df_shutuba = df_shutuba[df_shutuba['レースID'].astype(str).str[-2:] != '11']
# display(df_shutuba['レースID'])

# 辞書作成
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

rank_dict = {'1': 4, '2': 5, '3': 5, '4': 6, '5': 6, '6': 7, '7': 8, '8': 8, '9': 9, '10': 9, '11': 10, '12': 10, '13': 11, '14': 11, '15': 12,
             '16': 13, '17': 14, '18': 15}

dist_dict = {'-1600.0':     9.400000,
            '-1400.0':    13.666667,
            '-1300.0':     7.400000,
            '-1200.0':    10.375000,
            '-1100.0':    10.000000,
            '-1000.0':    11.264151,
            '-900.0':      9.947368,
            '-800.0':     10.379032,
            '-700.0':      9.853933,
            '-600.0':     10.635714,
            '-570.0':     18.000000,
            '-500.0':     10.266667,
            '-470.0':     14.000000,
            '-400.0':     10.932408,
            '-300.0':     11.123955,
            '-270.0':     10.888889,
            '-200.0':     11.672401,
            '-100.0':     11.276604,
            '-70.0':      11.000000,
            '0.0':       12.112223,
            '70.0':      20.000000,
            '100.0':     11.158619,
            '130.0':     20.000000,
            '150.0':      9.673611,
            '170.0':     10.000000,
            '200.0':     10.827936,
            '230.0':     11.000000,
            '250.0':      8.475610,
            '300.0':     11.047880,
            '350.0':      3.000000,
            '400.0':     10.015599,
            '450.0':      7.345238,
            '500.0':     10.399625,
            '530.0':      1.000000,
            '550.0':      3.000000,
            '600.0':      9.890603,
            '650.0':      4.000000,
            '700.0':      8.500000,
            '710.0':     10.500000,
            '800.0':      8.973684,
            '850.0':     11.000000,
            '900.0':      9.600000,
            '950.0':      8.000000,
            '1000.0':    10.123457,
            '1010.0':    7.500000,
            '1100.0':    14.700000,
            '1110.0':     8.500000,
            '1200.0':    10.693878,
            '1300.0':    12.360000,
            '1310.0':    10.500000,
            '1400.0':    10.021277,
            '1410.0':    20.000000,
            '1500.0':     9.200000,
            '1600.0':    11.242424,
            '1700.0':    10.000000,
            '1710.0':    16.000000,
            '1770.0':     6.000000,
            '1800.0':     9.757576,
            '1850.0':     6.000000,
            '1910.0':    12.000000,
            '2000.0':     9.600000}

place_dict ={'-1.0':     11.244439,
            '0.0':     11.832657,
            '1.0':     11.664892,
            '2.0':     11.704237,
            '3.0':     11.654097,
            '4.0':     11.285992,
            '5.0':      9.913314,
            '6.0':     11.068083,
            '7.0':     11.499323,
            '8.0':     10.666667,
            '10.0':     8.653061,
            '11.0':     9.052239,
            '12.0':     6.416667,
            '13.0':     8.709677,
            '14.0':     9.378571,
            '15.0':     9.744828,
            '16.0':     9.206897,
            '17.0':     7.358209,
            '18.0':     6.977011,
            '19.0':     8.597826,
            '20.0':     9.372881,
            '21.0':    12.714286,
            '22.0':     7.500000,
            '23.0':     7.136364}

field_dict = {'1.0':    11.606628, '2.0':    11.172710}

coner_dict = {'-9.0':     11.280740,
'-8.0':     11.550010,
'-7.0':     11.778557,
'-6.0':     11.825297,
'-5.0':     12.051354,
'-4.0':     12.182029,
'-3.0':     12.325500,
'-2.0':     12.058681,
'-1.0':     11.702875,
'1.0':      8.545455,
 '2.0':      7.000000,
 '3.0':      6.000000,
 '4.0':      7.000000,
 '5.0':      6.000000,
 '6.0':      8.642857,
 '7.0':      8.200000,
 '8.0':      5.461538,
 '9.0':     10.272727,
 '10.0':    10.963528,
 '11.0':    10.832201,
 '12.0':    10.662833,
 '13.0':    10.302537,
 '14.0':     9.954900,
 '15.0':     9.640118,
 '16.0':     9.359395,
 '17.0':     9.848276,
 '18.0':     9.333333}

jockey_dict = {'0':    12.627700, '1':    11.248748}


# カラム作成
df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

# コーパス作成
df_shutuba['文章リスト'] = df_shutuba['父馬'] + ' ' + '今走' + df_shutuba['レース名'] + df_shutuba['フィールド'] + df_shutuba['距離'].astype(str) + df_shutuba['馬場'] + ' ' + '前走' + df_shutuba['前走'].fillna('欠走') \
    + ' ' + '2走' + df_shutuba['2走'].fillna('欠走') + ' ' + '3走' + df_shutuba['3走'].fillna('欠走') + ' ' + '4走' + \
    + df_shutuba['4走'].fillna('欠走') + ' ' + '5走' + df_shutuba['5走'].fillna('欠走')

horse_mapping = {}
with open(horse_path, mode="rb") as f:
    horse_mapping = pickle.load(f)
with open(femal_horse_path, mode="rb") as f:
    femal_horse_mapping = pickle.load(f)
# df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
# df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

# いらないカラムを消す
df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', 'コーナー通過順', '厩舎', 'タイム', '騎手斤量', '着差', '後3F', '印'], axis=1)

df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')
# df['血統'] = pd.to_numeric((df['父馬'] * 10).astype(str) + df['母父馬'].astype(str), errors='coerce')

df['馬単'] = df['馬単'].str.replace(',', '')
df['馬単'] = df['馬単'].str.extract(r'(\d+円)', expand=True)
df['馬単'] = df['馬単'].str.extract(r'(\d+)', expand=True)

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

# 性別をマッピング
sex_mapping = {'牡': 1, '牝': 2, 'セ': 3}
df['性'] = df['性'].map(sex_mapping)

# 「性齢」「馬体重（増減）」はいらないので消す
df = df.drop(['性齢', '馬体重(増減)'], axis=1)

# 「前走」から必要なデータにわける
df_split = df['前走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
df_split.columns = ['日付', '1場所', '1過去着順', '1レース名', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', '1騎手', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差']

# 前走のカラムを削除
df = df.drop(['前走'], axis=1)
df_split = df_split.drop(['日付'], axis=1)

# 4角コーナー通過順のみに
df_split['1コーナー通過順'] = df_split['1コーナー通過順'].str[-4:-1].apply(lambda x : x if x != ' ' else None)
df_split['1コーナー通過順'] = df_split['1コーナー通過順'].astype(float).abs()

# クラス別に分類
df_split['1クラス'] = 0
class_dict = {'GI': 1, 'GII': 2, 'GIII': 3, 'OP': 4, 'L': 5, '3勝': 6, '1600万下': 6, '1600下': 6, '2勝': 7, '1000万下': 7, '1000下': 7,
              '1勝': 8, '500万下': 8, '500下': 8, '未勝利': 9}
for k, v in class_dict.items():
    df_split['1クラス'] = df_split['1クラス'].mask(df_split['1レース名'].str.contains(k, na=False), v)

# 文字列データを数値データにする
nagoya_mapping = {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
                  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
                  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}
df_split['1場所'] = df_split['1場所'].map(nagoya_mapping)
# df['場所'] = df['場所'].map(nagoya_mapping)

field_mapping = {'芝': 1, 'ダ': 2, '障': 3}
df_split['1フィールド'] = df_split['1フィールド'].map(field_mapping)
df['フィールド'] = df['フィールド'].map(field_mapping)

condition_mapping = {'良': 1, '稍': 2, '重': 3, '不': 4}
df_split['1馬場'] = df_split['1馬場'].map(condition_mapping)
df['馬場'] = df['馬場'].map(condition_mapping)

df_split['1着差'] = df_split['1着差'].astype(float) + (df_split['1クラス'].astype(int) * 0.5)

# 騎手のユニーク値から辞書をつくる
jockey_mapping = {}
with open(jockey_path, mode="rb") as f:
    jockey_mapping = pickle.load(f)

# 文字列から数値に変換する
df_split['1騎手'] = df_split['1騎手'].map(jockey_mapping)
df['騎手'] = df['騎手'].map(jockey_mapping)

with open(f'./pickle-dict/jwin_dict{field_num}.pkl', "rb") as dd:
    jwin = pickle.load(dd)

df['騎手'] = df['騎手'].map(jwin)
# print(df['騎手'])

# タイムを秒表記にする
base_time = pd.to_datetime('00:00.0', format='%M:%S.%f')

df_split['1タイム'] = pd.to_datetime(df_split['1タイム'], format='%M:%S.%f') - base_time
df_split['1タイム'] = df_split['1タイム'].dt.total_seconds()

# スピード指数の計算
for i in range(1, 11):
    for k in range(1000, 3700, 100):
        try:
            speed = speed_dict[f'{i}{k}1']
            df_split.loc[(df_split['1場所'] == i) & (df_split['1距離'].astype(float) == k) & (df_split['1フィールド'] == 1), '1スピード指数'] = ((speed + 0.01 - df_split['1タイム'].astype(float)) * 10) * (1 / speed * 100) + (df_split['1馬場'] * 10) + (df_split['1斤量'].astype(float) - 55) * 2 + 80
        except KeyError:
            pass
        try:
            speed = speed_dict[f'{i}{k}2']
            df_split.loc[(df_split['1場所'] == i) & (df_split['1距離'].astype(float) == k) & (df_split['1フィールド'] == 2), '1スピード指数'] = ((speed + 0.01 - df_split['1タイム'].astype(float)) * 10) * (1 / speed * 100) + (13 - (df_split['1馬場'] * 3)) + (df_split['1斤量'].astype(float) - 55) * 2 + 80
        except KeyError:
            pass

# 上がり3Fを指数化
df_split['1後3F'] = df_split['1後3F'].mask((df_split['1フィールド'] == 1) & df_split['1後3F'].notna() & df_split['1距離'].notna(), df_split['1後3F'].astype(float) / (0.94 + (df_split['1距離'].astype(float) / 20000)))
df_split['1後3F'] = df_split['1後3F'].mask((df_split['1フィールド'] == 2) & df_split['1後3F'].notna() & df_split['1距離'].notna(), df_split['1後3F'].astype(float) / (1.01 + (df_split['1距離'].astype(float) / 20000)))
df_split['1後3F'] = df_split['1後3F'].mask((df_split['1フィールド'] == 3) & df_split['1後3F'].notna() & df_split['1距離'].notna(), df_split['1後3F'].astype(float) / (0.36 + (df_split['1距離'].astype(float) * 1.5 / 100000)))

# 人気を裏切ったかどうか
# df_split['1人気'] = df_split['1人気'].replace('', np.nan)
# df_split['1過去着順'] = df_split['1過去着順'].replace('', np.nan)
# df_split['1出走馬数'] = df_split['1出走馬数'].replace('', np.nan)
# df_split['1人気差'] = float('nan')
# df_split['1人気差'] = df_split['1人気差'].mask(df_split['1人気'].notna() & df_split['1過去着順'].notna() & df_split['1出走馬数'].notna(), (df_split['1人気'].astype(float) - df_split['1過去着順'].astype(float)) / df_split['1出走馬数'].astype(float))

# df_split['人気増減'] = df_split['1人気'].astype(float) - df['人気'].astype(float)

# # 不要なカラムを削除
# df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
# df_split['1場所変化'] = 0
# df_split['1場所変化'] = df_split['1場所変化'].mask(df_split['1場所'] == field_num, 1)
# df_split['1フィールド変化'] = 0
# df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)

df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
df_split['1場所変化'] = df_split['1場所'] - field_num
df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

# df_split['1フィールド変化*スピード指数'] = df_split['1フィールド変化'] * df_split['1スピード指数']
# df_split['1場所変化*スピード指数'] = df_split['1場所変化'] * df_split['1スピード指数']
# df_split['1距離差*スピード指数'] = df_split['1距離差'] * df_split['1スピード指数']
# df_split['1コーナー*3F'] = df_split['1コーナー通過順'] * df_split['1後3F']
# df_split['馬番差'] = (df['出走頭数'].astype(float) / df['馬番'].astype(float)) - (df_split['1出走馬数'].astype(float) / df_split['1馬番'].astype(float))
# df_split['騎手変化'] = 1
# df_split['騎手変化'] = df_split['騎手変化'].mask(df_split['1騎手'] == df['騎手'], 0)
# df_split['騎手変化*スピード指数'] = df_split['騎手変化'] * df_split['1スピード指数']


# # df_split['1フィールド変化'] = 0
# # df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)

# df_split['1馬番差'] = (df['出走頭数'].astype(float) / df['馬番'].astype(float)) - (df_split['1出走馬数'].astype(float) / df_split['1馬番'].astype(float))
# df_split['1騎手変化'] = 1
# df_split['1騎手変化'] = df_split['1騎手変化'].mask(df_split['1騎手'] == df['騎手'], 0)


# df_split = df_split.drop(['1人気', '1レース名', '1タイム', '1騎手', '1出走馬数', '1馬番', '1斤量', '1馬体重', '1体重増減', '1距離','1騎手', '1馬場', '1フィールド', '1場所', '1クラス'], axis=1)
# df_split = df_split.drop(['1人気', '1レース名', '1タイム', '1出走馬数', '1馬番', '1斤量', '1馬体重', '1体重増減', '1距離','1騎手', '1馬場', '1フィールド', '1場所'], axis=1)
# df = df.drop(['騎手', '出走頭数', '性', '斤量', '馬場'], axis=1)
df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

# 今走と前走を結合
df_all = pd.concat([df, df_split], axis=1)

# target encording
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

# df_all = df_all.drop(['1クラス'], axis=1)
# df_all = df_all.drop(['平均クラス', '平均ペース'], axis=1)

######################################################################################
# 以降2~5走の処理
count = 2
sou = '2'
while True:
    # 「2走」から必要なデータにわける
    df_split = df[sou+'走'].astype(str).str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d).+(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
    df_split.columns = ['日付', sou+'場所', sou+'過去着順', sou+'レース名', sou+'フィールド', sou+'距離', sou+'タイム', sou+'馬場', sou+'出走馬数', sou+'馬番', sou+'人気', sou+'騎手', sou+'斤量', sou+'後3F', sou+'馬体重', sou+'体重増減', sou+'着差']

    # 2~5走のカラムを削除
    df_split = df_split.drop(['日付'], axis=1)

    # 文字列データを数値データにする
    df_split[sou+'場所'] = df_split[sou+'場所'].map(nagoya_mapping)

    df_split[sou+'フィールド'] = df_split[sou+'フィールド'].map(field_mapping)

    df_split[sou+'馬場'] = df_split[sou+'馬場'].map(condition_mapping)

    # 文字列から数値に変換する
    df_split[sou+'騎手'] = df_split[sou+'騎手'].map(jockey_mapping)

    # タイムを秒表記にする
    df_split[sou+'タイム'] = pd.to_datetime(df_split[sou+'タイム'], format='%M:%S.%f') - base_time
    df_split[sou+'タイム'] = df_split[sou+'タイム'].dt.total_seconds()

    # スピード指数の計算
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

    # 人気を裏切ったかどうか
    # df_split[sou+'人気'] = df_split[sou+'人気'].replace('', np.nan)
    # df_split[sou+'過去着順'] = df_split[sou+'過去着順'].replace('', np.nan)
    # df_split[sou+'出走馬数'] = df_split[sou+'出走馬数'].replace('', np.nan)
    # df_split[sou+'人気差'] = float('nan')
    # df_split[sou+'人気差'] = df_split[sou+'人気差'].mask(df_split[sou+'人気'].notna() & df_split[sou+'過去着順'].notna() & df_split[sou+'出走馬数'].notna(), (df_split[sou+'人気'].astype(float) - df_split[sou+'過去着順'].astype(float)) / df_split[sou+'出走馬数'].astype(float))

    # # 条件の変化
    # df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
    # df_split[sou+'場所変化'] = 0
    # df_split[sou+'場所変化'] = df_split[sou+'場所変化'].mask(df_split[sou+'場所'] == field, 1)
    # df_split[sou+'フィールド変化'] = 0
    # df_split[sou+'フィールド変化'] = df_split[sou+'フィールド変化'].mask(df_split[sou+'フィールド'] == df['フィールド'], 1)

    # 条件の変化
    # df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
    # df_split[sou+'場所変化'] = df_split[sou+'場所'] - field_num
    # df_split[sou+'フィールド変化'] = df_split[sou+'フィールド'] - df['フィールド']
    # # df_split['1フィールド変化'] = 0
    # # df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)
    # df_split[sou+'馬番差'] = (df['出走頭数'].astype(float) / df['馬番'].astype(float)) - (df_split[sou+'出走馬数'].astype(float) / df_split[sou+'馬番'].astype(float))
    # df_split[sou+'騎手変化'] = 1
    # df_split[sou+'騎手変化'] = df_split[sou+'騎手変化'].mask(df_split[sou+'騎手'] == df['騎手'], 0)

    # 不要なカラムを削除
    # df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)
    df_split = df_split.drop([sou+'レース名', sou+'騎手'], axis=1)

    # 今走と過去走を結合
    df_all = pd.concat([df_all, df_split], axis=1)

    if count == 5:
        break
    elif count == 2:
        sou = '3'
    elif count == 3:
        sou = '4'
    else:
        sou = '5'
    count += 1

# 表現学習
# vecter = []
# m = Doc2Vec.load(fname)

# for i in range(len(df_all.index)):
#     vecter.append(m.infer_vector(df_all.iloc[i, df_all.columns.get_loc('文章リスト')].split(), epochs=20))

# vecter = pd.DataFrame(vecter)

# df_all = pd.concat([df_all.reset_index(drop=True), vecter.reset_index(drop=True)], axis=1)
# print(df_all.isnull().sum())

# Target Encording
# for i in range(2, 6):
#     with open(f'./pickle-dict/{i}class_dict{field_num}.pkl', "rb") as dd:
#         class_dict = pickle.load(dd)
#     with open(f'./pickle-dict/{i}kyori_dict{field_num}.pkl', "rb") as dd:
#         kyori_dict = pickle.load(dd)
#     with open(f'./pickle-dict/{i}basyo_dict{field_num}.pkl', "rb") as dd:
#         basyo_dict = pickle.load(dd)
#     with open(f'./pickle-dict/{i}field_dict{field_num}.pkl', "rb") as dd:
#         field_dict = pickle.load(dd)
#     with open(f'./pickle-dict/{i}ninkisa_dict{field_num}.pkl', "rb") as dd:
#         ninki_dict = pickle.load(dd)
#     with open(f'./pickle-dict/{i}jhenka_dict{field_num}.pkl', "rb") as dd:
#         jockey_dict = pickle.load(dd)
    
#     df_all[f'{i}クラス差'] = pd.to_numeric(df_all[f'{i}クラス差'].astype(float).map(class_dict), errors='coerce')
#     df_all[f'{i}距離差'] = pd.to_numeric(df_all[f'{i}距離差'].astype(float).map(kyori_dict), errors='coerce')
#     df_all[f'{i}場所変化'] = pd.to_numeric(df_all[f'{i}場所変化'].astype(float).map(basyo_dict), errors='coerce')
#     df_all[f'{i}フィールド変化'] = pd.to_numeric(df_all[f'{i}フィールド変化'].astype(float).map(field_dict), errors='coerce')
#     df_all[f'{i}騎手変化'] = pd.to_numeric(df_all[f'{i}騎手変化'].astype(float).map(jockey_dict), errors='coerce')

# with open(f'./pickle-dict/up_dict{field_num}.pkl', "rb") as dd:
#     up_dict = pickle.load(dd)
# df_all[f'上昇度'] = pd.to_numeric(df_all[f'上昇度'].astype(float).map(up_dict), errors='coerce')

# Target Encording
# df_all['2騎手変化'] = 1
# df_all['2騎手変化'] = df_all['騎手変化'].mask(df_all['2騎手'] == df_all['騎手'], 0)

# print(df_all[['1距離差', '1場所変化', '1フィールド変化', '1コーナー通過順', '騎手変化']].isnull().sum())
# df_all.fillna({f'1距離差': df_all[f'2距離差'], f'1場所変化': df_all['2場所'] - field_num, f'1フィールド変化': df_all['2フィールド'] - df_all['フィールド'], '騎手変化': df_all['2騎手変化']}, inplace=True)
# print(df_all[['1距離差', '1場所変化', '1フィールド変化', '1コーナー通過順', '騎手変化']].isnull().sum())

# with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
#     dist_dict = pickle.load(dd)
# with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
#     place_dict = pickle.load(dd)
# with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
#     field_dict = pickle.load(dd)
# # # with open(f'./pickle-dict/corner_dict{field_num}.pkl', "rb") as dd:
# # #     coner_dict = pickle.load(dd)
# # with open(f'./pickle-dict/jhenka_dict{field_num}.pkl', "rb") as dd:
# #     jockey_dict = pickle.load(dd)

# # with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
# #     field_dict = pickle.load(dd)
# # # with open(f'./pickle-dict/1ninkisa_dict{field_num}.pkl', "rb") as dd:
# # #     ninki_dict = pickle.load(dd)
# # # with open(f'./pickle-dict/jokey_leading_dict{field_num}.pkl', "rb") as dd:
# # #     leading_dict = pickle.load(dd)
# # # with open(f'./pickle-dict/brad_dict{field_num}.pkl', "rb") as dd:
# # #     brad_dict = pickle.load(dd)
# with open(f'./pickle-dict/1class_dict{field_num}.pkl', "rb") as dd:
#     classsa_dict = pickle.load(dd)
with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
    sire_dict = pickle.load(dd)
# with open(f'./pickle-dict/bms_dict{field_num}.pkl', "rb") as dd:
#     bms_dict = pickle.load(dd)
# with open(f'./pickle-dict/ninkizougen_dict{field_num}.pkl', "rb") as dd:
#     ninkizougen_dict = pickle.load(dd)
# # with open(f'./pickle-dict/yuri_umaban_dict{field_num}.pkl', "rb") as dd:
# #     yuri_dict = pickle.load(dd)

# # df_all['有利馬番'] = (df_all['距離'] * 10) + (df_all['馬番'] * 10) + df_all['フィールド']
# # df_all['有利馬番'] = pd.to_numeric(df_all['有利馬番'].astype(float).map(yuri_dict), errors='coerce')
# # # display(df_all[['1距離差', '1場所変化', '1フィールド', '1コーナー通過順', '騎手変化']])
# # df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
# # df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
# # df_all['1フィールド'] = pd.to_numeric(df_all['1フィールド'].astype(float).map(field_dict), errors='coerce')
# # df_all['1コーナー通過順'] = pd.to_numeric(df_all['1コーナー通過順'].astype(float).map(coner_dict), errors='coerce')
# # df_all['騎手変化'] = pd.to_numeric(df_all['騎手変化'].astype(float).map(jockey_dict), errors='coerce')
# # print(df_all[['1距離差', '1場所変化', '1フィールド変化', '1コーナー通過順', '1騎手変化', '1スピード指数', '1人気差', '騎手',  '1クラス差', '父馬', '母父馬']].isnull().sum())

# df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
# df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
# # # # # df_all['1コーナー通過順'] = pd.to_numeric(df_all['1コーナー通過順'].astype(float).map(coner_dict), errors='coerce')
# # # # df_all['1騎手変化'] = pd.to_numeric(df_all['1騎手変化'].astype(float).map(jockey_dict), errors='coerce')

# df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
# # # df_all['1人気差'] = pd.to_numeric(df_all['1人気差'].astype(float).map(ninki_dict), errors='coerce')
# # # df_all['騎手'] = pd.to_numeric(df_all['騎手'].astype(float).map(leading_dict), errors='coerce')
# # # df_all['血統'] = pd.to_numeric(df_all['血統'].astype(str).map(brad_dict), errors='coerce')
# df_all['1クラス差'] = pd.to_numeric(df_all['1クラス差'].astype(float).map(classsa_dict), errors='coerce')
df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')
# df_all['母父馬'] = pd.to_numeric(df_all['母父馬'].astype(float).map(bms_dict), errors='coerce')
# df_all['人気増減'] = pd.to_numeric(df_all['人気増減'].astype(float).map(ninkizougen_dict), errors='coerce')

# print(df_all.isnull().sum())

# display(df_all[['父馬', '騎手', '母父馬','1距離差','1コーナー通過順','1クラス差','1フィールド変化','1場所変化', '1スピード指数']].head(40))

# display(df_all[['1距離差', '1場所変化', '1フィールド', '1コーナー通過順', '騎手変化']])


# 着順から文字列を排除
indexNames = df_all[(df_all['着順'] != '中止') & (df_all['着順'] != '除外') & (df_all['着順'] != '取消') & (df_all['着順'] != '失格') & (df_all['着順'] != '未定')]
df_all = indexNames

df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

# df_all から Nan を各列の中央値に置換する 4用
# df_all = df_all.drop(['2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '父馬', '母父馬', '1コーナー通過順'], axis=1)

# 5用
#df_all = df_all.drop(['2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '父馬', '母父馬', '1コーナー通過順', '間隔','1着差', '2着差', '3着差', '4着差', '5着差','1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数'], axis=1)

# 6用
# df_all = df_all.drop(['2走', '3走', '4走', '5走', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '馬場', '出走頭数', '間隔','1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数', '平均クラス'], axis=1)
# df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬', '騎手', '1コーナー通過順', 'レース名', '文章リスト'], axis=1)
# display(df_all.columns.values)
# df_all = df_all.drop(['2走', '3走', '4走', '5走', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '馬場', '出走頭数', '父馬', '母父馬', '間隔','1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数'], axis=1)


# # df_all から Nan を各列の中央値に置換、削除する 3用
# df_all = df_all.drop(['2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順',], axis=1)

# list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気', '騎手', '斤量', '後3F', '馬体重', '体重増減', '着差', '人気差', 'スピード指数']

df_all = df_all.drop(['2走', '3走', '4走', '5走',  '母父馬', 'レース名',  '文章リスト'], axis=1)

list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気',  '斤量', '後3F', '馬体重', '体重増減', '着差', 'スピード指数']

# for i in range(1,5):
#     k = i + 1
#     for v in list_columns:
#         df_all.fillna({f'{i}{v}': df_all[f'{k}{v}']}, inplace=True)
#         df_all.fillna({f'{k}{v}': df_all[f'{i}{v}']}, inplace=True)
    
# 空白削除
for i in df_all.columns:
    df_all[i] = df_all[i].replace('', np.nan)

# 上昇度カラム作成
df_all['5過去着順'] = df_all['5過去着順'].astype(float)
df_all['4過去着順'] = df_all['4過去着順'].astype(float)
df_all['3過去着順'] = df_all['3過去着順'].astype(float)
df_all['2過去着順'] = df_all['2過去着順'].astype(float)
df_all['1過去着順'] = df_all['1過去着順'].astype(float)
# df_all['上昇度'] = (df_all['5過去着順'] - df_all['4過去着順']) + (df_all['4過去着順'] - df_all['3過去着順']) + (df_all['3過去着順'] - df_all['2過去着順']) + (df_all['2過去着順'] - df_all['1過去着順']) / (df_all[['1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順']].isnull().sum(axis=1) + 1)
# df_all['上昇度'] = np.sign(df_all['5過去着順'] - df_all['4過去着順']) + np.sign(df_all['4過去着順'] - df_all['3過去着順']) + np.sign(df_all['3過去着順'] - df_all['2過去着順']) + np.sign(df_all['2過去着順'] - df_all['1過去着順'])

# with open(f'./pickle-dict/up_dict{field_num}.pkl', "rb") as dd:
#     up_dict = pickle.load(dd)
# df_all['上昇度'] = pd.to_numeric(df_all['上昇度'].astype(float).map(up_dict), errors='coerce')
print(df_all.isnull().sum())
# d = df_all.groupby(f'上昇度')['rank'].mean()
# dict = d.to_dict()
# with open(f'./pickle-dict/up_dict{field}.pkl', "wb") as dd:
#     pickle.dump(dict, dd)
# df_all = Target_encording(df_all, '上昇度', 'rank')


# レース内の順序をシャッフル
df_all = df_all.sample(frac=1)
df_all = df_all.sort_values(['レースID'], ascending=True)

# 1着のみ単勝オッズを保有
df_all['オッズ'] = df_all['単勝オッズ']
df_all['単勝オッズ'] = df_all['単勝オッズ'] * 100
df_all['単勝オッズ'] = df_all['単勝オッズ'].where(df_all['着順'].astype(int) == 1, 0)

# カテゴリー変数の準備
cat_list = ['馬番','フィールド','馬場','性','1場所','1フィールド','1馬場','1馬番','1コーナー通過順','2場所','2フィールド','2馬場','2馬番','3場所','3フィールド','3馬場','3馬番',
            '4場所','4フィールド','4馬場','4馬番','5場所','5フィールド','5馬場','5馬番']

# cat_list = ['父馬','騎手','馬番','フィールド','馬場','性','1場所','1フィールド','1馬場','1馬番','1コーナー通過順','2場所','2フィールド','2馬場','2馬番','3場所','3フィールド','3馬場','3馬番',
#             '4場所','4フィールド','4馬場','4馬番','5場所','5フィールド','5馬場','5馬番']

# df_all.loc[:, cat_list] = df_all.loc[:, cat_list].astype(float).apply(lambda x: x - 1)
# df_all['1場所変化'] = df_all['1場所変化'] + 1
# df_all['1フィールド変化'] = df_all['1フィールド変化'] + 2
# df_all['齢'] = df_all['齢'].astype(float) - 2

# dist_list = ['距離','1距離','2距離','3距離','4距離','5距離']

# df_all.loc[:, dist_list] = df_all.loc[:,dist_list].mask(df_all.loc[:,dist_list].astype(float) > 2700, 4)
# df_all.loc[:,dist_list] = df_all.loc[:,dist_list].mask(df_all.loc[:,dist_list].astype(float) > 2100, 3)
# df_all.loc[:,dist_list] = df_all.loc[:,dist_list].mask(df_all.loc[:,dist_list].astype(float) >= 1900, 2)
# df_all.loc[:,dist_list] = df_all.loc[:,dist_list].mask(df_all.loc[:,dist_list].astype(float) > 1300, 1)
# df_all.loc[:,dist_list] = df_all.loc[:,dist_list].mask(df_all.loc[:,dist_list].astype(float) >= 800, 0)

# categorical_featureリスト
# cat_list = ['父馬','騎手','馬番','距離','フィールド', '馬場','性', '齢','1場所','1フィールド','1距離','1馬場','1コーナー通過順','1クラス','1馬番', '1場所変化', '1フィールド変化',
# '2場所','2フィールド','2距離','2クラス','2馬場','2馬番',
# '3場所','3フィールド','3距離','3クラス','3馬場','3馬番',
# '4場所','4フィールド','4距離','4クラス','4馬場','4馬番',
# '5場所','5フィールド','5距離','5クラス','5馬場','5馬番']

cat_list = ['馬番','距離','フィールド', '馬場','性', '齢','1場所','1フィールド','1距離','1馬場','1コーナー通過順','1クラス','1馬番','1場所変化','1フィールド変化',
'2場所','2フィールド','2距離','2クラス','2馬場','2馬番',
'3場所','3フィールド','3距離','3クラス','3馬場','3馬番',
'4場所','4フィールド','4距離','4クラス','4馬場','4馬番',
'5場所','5フィールド','5距離','5クラス','5馬場','5馬番']

# 空白削除
for i in df_all.columns:
    df_all[i] = df_all[i].replace('', np.nan)

# 2023のデータを分離
SplitYear = 202200000000
eval_year = 202100000000
df_test = df_all.copy()
# df_test = df_all[df_all['レースID'] >= SplitYear]
# df_all = df_all[df_all['レースID'] < SplitYear]
df_eval = df_all[df_all['レースID'] >= eval_year]
# df_all = df_all[df_all['レースID'] < eval_year]

# 説明変数,目的変数
# , '人気', 'オッズ'
X = df_all.drop(['着順', '単勝オッズ', 'オッズ', '馬単'], axis=1)
X2 = df_test.drop(['着順', 'オッズ', '単勝オッズ', '馬単'], axis=1)
y = df_all[[ 'レースID', '着順', '単勝オッズ', '馬単']]
y2 = df_test[[ 'レースID', '着順', '単勝オッズ', '馬単', 'オッズ']]

# train, eval, testに分割
X_train = X
y_train = y
X_eval = df_eval.drop(['着順',  'オッズ', '単勝オッズ', '馬単'], axis=1)
# X_eval = df_eval.drop(['着順',  '人気', '単勝オッズ', '馬単'], axis=1)
y_eval = df_eval[[ 'レースID', '着順', '単勝オッズ', '馬単']]
X_test = X2
y_test = y2
# display(X_test['レースID'].head(40))
# クエリListを作成
id_count = X_train['レースID'].value_counts(sort=False)
train_list = id_count.values.tolist()

id_count = X_test['レースID'].value_counts(sort=False)
test_list = id_count.values.tolist()

id_count = X_eval['レースID'].value_counts(sort=False)
eval_list = id_count.values.tolist()

id_count = X2['レースID'].value_counts(sort=False)
X2_list = id_count.values.tolist()

# 検証用のレースIDを保存
y_test_id = pd.DataFrame()
stack = pd.DataFrame()
y_test_id2 = pd.DataFrame()
y_test_id['レースID'] = y_test['レースID']
y_test_id['着順'] = y_test['着順']
y_test_id['単勝オッズ'] = y_test['単勝オッズ']
y_test_id['馬単'] = y_test['馬単'].astype(int)
y_test_id['オッズ'] = y_test['オッズ']

y_test_id2['レースID'] = y2['レースID']
y_test_id2['着順'] = y2['着順']
y_test_id2['単勝オッズ'] = y2['単勝オッズ']
y_test_id2['馬単'] = y2['馬単'].astype(int)

# レースIDカラムを削除
X_train = X_train.drop(['レースID'], axis=1)
X_test = X_test.drop(['レースID'], axis=1)
X_eval = X_eval.drop(['レースID'], axis=1)
X2 = X2.drop(['レースID'], axis=1)
y_train = y_train.drop(['レースID'], axis=1)
y_train = y_train.drop(['着順'], axis=1)
y_train = y_train.drop(['単勝オッズ'], axis=1)
y_train = y_train.drop(['馬単'], axis=1)
y_test = y_test.drop(['レースID'], axis=1)
y_test = y_test.drop(['着順'], axis=1)
y_test = y_test.drop(['単勝オッズ'], axis=1)
y_test = y_test.drop(['馬単'], axis=1)

y_eval = y_eval.drop(['レースID'], axis=1)
y_eval = y_eval.drop(['着順'], axis=1)
y_eval = y_eval.drop(['単勝オッズ'], axis=1)
y_eval = y_eval.drop(['馬単'], axis=1)
y2 = y2.drop(['レースID'], axis=1)
y2 = y2.drop(['着順'], axis=1)
y2 = y2.drop(['単勝オッズ'], axis=1)
y2 = y2.drop(['馬単'], axis=1)

# dataframeを値のみに
print(list(X_train.columns.values))
X_train = X_train.values
X_test = X_test.values
X_eval = X_eval.values
X2 = X2.values
y_train = y_train.values
y_test = y_test.values
y_eval = y_eval.values
y2 = y2.values

for i in range(1, file_num+1):
    # モデル呼び出し
    with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
        model = pickle.load(f)

    # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
    y_pred = model.predict(X_test, group=test_list)
    y_test_id[f'result{i}'] = y_pred

print(model.params)

# 説明変数追加
temp = 0
Z = y_test_id.iloc[:, 5:]
Z['Average'] = Z.iloc[:, :].mean(axis=1)
Z['レースID'] = y_test_id['レースID']
Z['単勝オッズ'] = y_test_id['単勝オッズ']
Z['オッズ'] = y_test_id['オッズ']
Z['着順'] = y_test_id['着順']
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

result = pd.DataFrame()
result['単勝オッズ'] = Z['単勝オッズ']
result['オッズ'] = Z['オッズ']
result['レースID'] = Z['レースID']
tya = Z['着順']

# レースIDカラムを削除
Z = Z.drop(['レースID', '単勝オッズ', 'オッズ', '着順'], axis=1)

# カラム名削除
Z = Z.values

for i in range(1, file_num+1):
    # モデル呼び出し
    with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
        model = pickle.load(f)

    # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
    y_pred = model.predict(Z, group=train2_list)
    result[f'result{i}'] = y_pred


y_test_id['Average'] = y_test_id.iloc[:, 5:].mean(axis=1)
y_test_id = y_test_id.sort_values(['レースID', 'Average'], ascending=[True, False])

result['Average'] = result.iloc[:, 3:].mean(axis=1)
result = result.sort_values(['レースID', 'Average'], ascending=[True, False])

# 予測スコアを標準化
counts = 0
mean_df = y_test_id['Average'].mean()
std_df = y_test_id['Average'].std()
y_test_id['Average'] = (y_test_id['Average'] - mean_df) / std_df

print(mean_df, std_df)

# counts = 0
# mean_df = result['Average'].mean()
# std_df = result['Average'].std()
# result['Average'] = (result['Average'] - mean_df) / std_df
# print(mean_df, std_df)

n_list = train2_list[:-1]
n = 0
for i in n_list:
    n += i
    mean_df = result.iloc[n-i:n, result.columns.get_loc('Average')].mean()
    std_df = result.iloc[n-i:n, result.columns.get_loc('Average')].std()
    result.iloc[n-i:n, result.columns.get_loc('Average')] = (result.iloc[n-i:n, result.columns.get_loc('Average')] - mean_df) / std_df
    

# 回収率計算
plotlist = []
holdlist = []
countlist = []
plotlist2 = []
countlist2 = []
holdlist2 = []
a = 0
b = 0
c = []
# 単勝
for k in range(0, 31):
    c = []
    count = 0
    num = 0
    sum = 0
    threshold = k / 10
    count_ = 0
    sum_ = 0
    new_list = test_list[:-1]
    for i in new_list:
        # 1着予想馬と2着予想馬のスコア差で判定
        if ((y_test_id.iloc[num, -1].astype(float) - y_test_id.iloc[num+1, -1].astype(float))) >= threshold:
            sum += (y_test_id.iloc[num, 2]).astype(float)
            # if y_test_id.iloc[num, 2].astype(float) != 0:
            #     c.append(y_test_id.iloc[num, 2].astype(float))
            #     if y_test_id.iloc[num, 2].astype(float) < 400:
            #         a += 1
            #     else:
            #         b += 1
            #     sum += 1
            count += 1
        # 1着予想馬のスコアのみで判定
        if ((y_test_id.iloc[num, -1].astype(float))) >= threshold:
            sum_ += (y_test_id.iloc[num, 2]).astype(float)
            # if y_test_id.iloc[num, 2].astype(float) != 0:
            #     sum_ += 1
            count_ += 1
        num += i
    try:
        countlist.append(count)
        plotlist.append(sum / (count))
        holdlist.append(threshold)
    except:
        pass
    try:
        countlist2.append(count_)
        plotlist2.append(sum_ / (count_))
        holdlist2.append(threshold)
    except:
        pass
    # print(a,b)
    # print(stats.hmean(c))
print(f"各閾値ごとの買い目点数(スコア差)\n{countlist}")
print(f"各閾値ごとの回収率(スコア差)\n{plotlist}")
plt.title('単勝(スコア差)')
plt.scatter(holdlist, plotlist)
plt.show()
print(f"各閾値ごとの買い目点数(スコアのみ)\n{countlist2}")
print(f"各閾値ごとの回収率(スコアのみ)\n{plotlist2}")
plt.title('単勝(スコアのみ)')
plt.scatter(holdlist2, plotlist2)
plt.show()

# # 上位3頭のみのdata
# result['着順'] = tya
n_list = train2_list[:-1]
n = 0
new_df = pd.DataFrame()
for i in n_list:
    n += i
    new_df = pd.concat([new_df, result.iloc[n-i:n-i+1, :]], axis=0)
result = new_df
# # result.to_csv("./csv/test_2024+2age.csv", na_rep='NaN')
# 56-75, 1.8-3.7

hoge = []
copy = result.copy()
# copy = copy[copy['Average'] >= 1.2]
rank = copy['Average'].map('{:.1f}'.format).value_counts()
rank = rank.sort_index()
copy['Average'] = copy['Average'].map('{:.1f}'.format)

# # print(len(result[result['着順'].astype(float) <= 3]) / len(copy))

# # print(copy['単勝オッズ'].sum() / (len(copy) * 100),stats.hmean(copy[copy['単勝オッズ'] != 0]['単勝オッズ'].tolist()))

pre_d = {}
copy['単勝オッズ'] = copy['単勝オッズ'].mask(copy['単勝オッズ'] != 0, 1)
# temp_df = copy.copy()
# temp_df = temp_df[:int(len(temp_df) * 0.5)]
for i in range(-38, 60):
    i = float(Decimal(str(i)) * Decimal('0.1'))
    # copy2 = temp_df[temp_df['Average'].astype(float) >= i]
    copy2 = copy[copy['Average'].astype(float) >= i]
    pre_d[i] = copy2['単勝オッズ'].sum() / len(copy2)
    # print(copy2['単勝オッズ'].sum() / len(copy2))

# display(pre_d)
with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "wb") as dd:
    pickle.dump(pre_d, dd)

# copy['単勝オッズ'] = copy['単勝オッズ'].mask(copy['単勝オッズ'] != 0, 1)
av = copy.groupby('Average')['単勝オッズ'].sum()

# # print(copy['単勝オッズ'].sum() / len(copy))

# # display(av)
display(rank)
for r, a in zip(rank, av):
    hoge.append(a / r)
# print(hoge)

for i, v in zip(av.index, hoge):
    print(f'{i}: {v}')

# print(pre_d)
result['Average'] = result['Average'].map('{:.1f}'.format)
result['Average'] = result['Average'].astype(float).map(pre_d)
# display(result['Average'].head(20))
result['ex'] = result['Average'] * result['オッズ']
# result['ex'] = 0
# for i, l in zip(av.index, hoge):
#     result['ex'] = result['ex'].mask(result['Average'].map('{:.1f}'.format) == i, l * (result['オッズ'] / 100))

# result = result[int(len(result) * 0.5):]
for i in range(0, 50):
    try:
        ex = result[result['ex'] >= float(Decimal(str(i)) * Decimal('0.1'))]
        # ex = result[result['ex'] >= i]
        # ex = ex[ex['ex'] < 1.9]
        print(i, ex['単勝オッズ'].sum(), (len(ex) * 100), ex['単勝オッズ'].sum() / (len(ex) * 100))
        # display(ex['レースID'][ex['単勝オッズ'] != 0].tail(5))

        print((stats.hmean(ex[ex['単勝オッズ'] != 0]['単勝オッズ'].tolist())), len(ex[ex['単勝オッズ'] != 0]) / len(ex))
    except:
        continue


# 回収率計算
plotlist = []
holdlist = []
countlist = []
plotlist2 = []
countlist2 = []
holdlist2 = []
a = 0
b = 0
c = []
# 単勝
for k in range(0, 31):
    c = []
    count = 0
    num = 0
    sum = 0
    threshold = k / 10
    count_ = 0
    sum_ = 0
    new_list = train2_list[:-1]
    for i in new_list:
        # 1着予想馬と2着予想馬のスコア差で判定
        if ((result.iloc[num, -1].astype(float) - result.iloc[num+1, -1].astype(float))) >= threshold:
            # if ((y_test_id.iloc[num, -1].astype(float))) <= 1:
            # if result.iloc[num, result.columns.get_loc('オッズ')].astype(float) >= 400:
            # sum += (result.iloc[num, 0]).astype(float)
            if y_test_id.iloc[num, 2].astype(float) != 0:
                # c.append(y_test_id.iloc[num, 2].astype(float))
            #     if y_test_id.iloc[num, 2].astype(float) < 400:
            #         a += 1
            #     else:
            #         b += 1
                sum += 1
            count += 1
        # 1着予想馬のスコアのみで判定
        if ((y_test_id.iloc[num, -1].astype(float))) >= threshold:
            # sum_ += (result.iloc[num, 0]).astype(float)
            if y_test_id.iloc[num, 2].astype(float) != 0:
                sum_ += 1
            count_ += 1
        num += i
    try:
        countlist.append(count)
        plotlist.append(sum / (count))
        holdlist.append(threshold)
    except:
        pass
    try:
        countlist2.append(count_)
        plotlist2.append(sum_ / (count_))
        holdlist2.append(threshold)
    except:
        pass
    # print(a,b)
    # print(stats.hmean(c))
print(f"各閾値ごとの買い目点数(スコア差)\n{countlist}")
print(f"各閾値ごとの回収率(スコア差)\n{plotlist}")
plt.title('単勝(スコア差)')
plt.scatter(holdlist, plotlist)
plt.show()
print(f"各閾値ごとの買い目点数(スコアのみ)\n{countlist2}")
print(f"各閾値ごとの回収率(スコアのみ)\n{plotlist2}")
plt.title('単勝(スコアのみ)')
plt.scatter(holdlist2, plotlist2)
plt.show()
