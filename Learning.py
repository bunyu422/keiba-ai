import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
import pickle
from IPython.display import display
import optuna.integration.lightgbm as lgb
import lightgbm as lgbm
import time
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import gc
import warnings

warnings.simplefilter('ignore')

# 開催場所番号
field = 2

# {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
#  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
#  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

# 学習済みモデルを保存するファイルネーム
file_name = ""

# ファイルパス
csv_path = "./csv/tokyo_2012-2023.csv" # 学習に使うcsvデータのパス
horse_path = "./pickle-dict/horse_jra.pkl" # 父馬のマッピング用辞書のパス
femal_horse_path = "./pickle-dict/femal_horse_jra.pkl" # 母父馬のマッピング用辞書のバス
jockey_path = "./pickle-dict/jockey_jra.pkl" # 騎手のマッピング用辞書のパス
tuner_path = f"./pickle-tuner/{file_name}" # 学習済みモデルを保存する場所

# データフレーム生成
df = pd.DataFrame()

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)

# 列を全表示（列の数）
pd.set_option("display.max_columns", None)

# csvファイル読み込み(スクレイピングしない場合)
df = pd.read_csv(csv_path, index_col=0)

# スクレイピング
# for year in range(2012, 2024):
#     for number in range(1, 6):
#         for day in range(1, 10):
#             for race_no in range(1, 13):
#                 race_id = '{}06{}{}{}'.format(str(year), str(number).zfill(2), str(day).zfill(2), str(race_no).zfill(2))
#                 url_race = 'https://race.netkeiba.com/race/result.html?race_id={}&rf=race_list'.format(race_id)
#                 url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
#                 try:
#                     df_result = pd.read_html(url_race)[0]
#                     df_past = pd.read_html(url_past)[0]
#                     r = requests.get(url_race)
#                     soup = BeautifulSoup(r.content, 'html.parser')
#                     data1 = soup.find('div', class_='RaceData01').text
#                     data2 = soup.find('div', class_='RaceData02').text
#                     data3 = soup.find('tr', class_='Umatan').text
#                     a = data2[data2.find('新馬')+0: data2.find('新馬')+2]
#                     # if int(data2[data2.find('サラ系')+3: data2.find('サラ系')+4]) == 2:
#                     #     continue
#                     if a == '新馬':
#                         continue
#                     df_result_past = pd.merge(df_result, df_past, on='馬番')
#                     df_result_past['距離'] = re.findall(r'\d+', data1)[2]
#                     df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
#                     df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
#                     df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]
#                     df_result_past['馬単'] = data3
#                     print(url_race)
#                     time.sleep(1)
#                 except:
#                     continue
#                 df_result_past['レースID'] = race_id
#                 df = pd.concat([df, df_result_past])

# # 結果をcsvに保存
# df.to_csv(csv_path, na_rep='NaN')

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

df = df.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し

# 新たなカラムを作成
df['父馬'] = df['馬名_y'].str.extract(r'(\w+\s)', expand=True)
df['間隔'] = df['馬名_y'].str.extract(r'(\d+)', expand=True)
df['母父馬'] = df['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

# 血統pickle作成
# horse_mapping = dict(zip(df['父馬'].unique().tolist(), range(1, len(df['父馬'].unique().tolist()) + 1)))
# with open(horse_path, "wb") as jd:
#     pickle.dump(horse_mapping, jd)
# femal_house_mapping = dict(zip(df['母父馬'].unique().tolist(), range(1, len(df['父馬'].unique().tolist()) + 1)))
# with open("femal_horse_path", "wb") as jd:
#      pickle.dump(femal_house_mapping, jd)

# 血統pickle呼び出し
with open(horse_path, mode="rb") as f:
    horse_mapping = pickle.load(f)
with open(femal_horse_path, mode="rb") as f:
    femal_horse_mapping = pickle.load(f)

# マッピング
df['父馬'] = df['父馬'].map(horse_mapping)
df['母父馬'] = df['母父馬'].map(femal_horse_mapping)

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

# 「性齢」「馬体重（増減）」はいらないので消す
df = df.drop(['性齢', '馬体重(増減)', '馬体重', '体重増減'], axis=1)

# 「前走」から必要なデータにわける
df_split = df['前走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
df_split.columns = ['日付', '1場所', '1過去着順', '1レース名', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', '1騎手', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差']

# 前走のカラムを削除
df = df.drop(['前走'], axis=1)
df_split = df_split.drop(['日付'], axis=1)

# 4角コーナー通過順のみに
df_split['1コーナー通過順'] = df_split['1コーナー通過順'].str[-4:-1].astype(float)

# クラス別に分類
df_split['1クラス'] = 0
class_dict = {'GI': 1, 'GII': 2, 'GIII': 3, 'OP': 4, 'L': 4, '3勝': 5, '1600万下': 5, '1600下': 5, '2勝': 6, '1000万下': 6, '1000下': 6,
              '1勝': 7, '500万下': 7, '500下': 7, '未勝利': 8, '新馬': 9}
for k, v in class_dict.items():
    df_split['1クラス'] = df_split['1クラス'].mask(df_split['1レース名'].str.contains(k, na=False), v)

# 文字列データを数値データにする
nagoya_mapping = {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
                  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
                  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}
df_split['1場所'] = df_split['1場所'].map(nagoya_mapping)

field_mapping = {'芝': 1, 'ダ': 2, '障': 3}
df_split['1フィールド'] = df_split['1フィールド'].map(field_mapping)
df['フィールド'] = df['フィールド'].map(field_mapping)

condition_mapping = {'良': 1, '稍': 2, '重': 3, '不': 4}
df_split['1馬場'] = df_split['1馬場'].map(condition_mapping)
df['馬場'] = df['馬場'].map(condition_mapping)

df_split['1着差'] = df_split['1着差'].astype(float) + (df_split['1クラス'].astype(int) * 0.3)

# 騎手のユニーク値から辞書をつくる
jockey_mapping = {}

# 騎手pickle呼び出し
with open(jockey_path, mode="rb") as f:
    jockey_mapping = pickle.load(f)

# 既存の騎手リストに追加する場合
# for name, num in zip(df['騎手'].unique().tolist(), range(1, len(df['騎手'].unique().tolist()) + 1)):
#     jockey_mapping.setdefault(name, num)

# 騎手pickle保存
# with open(jockey_path, "wb") as jd:
#     pickle.dump(jockey_mapping, jd)

# 文字列から数値に変換する
df_split['1騎手'] = df_split['1騎手'].map(jockey_mapping)
df['騎手'] = df['騎手'].map(jockey_mapping)


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
df_split['1人気'] = df_split['1人気'].replace('', np.nan)
df_split['1過去着順'] = df_split['1過去着順'].replace('', np.nan)
df_split['1出走馬数'] = df_split['1出走馬数'].replace('', np.nan)
df_split['1人気差'] = float('nan')
df_split['1人気差'] = df_split['1人気差'].mask(df_split['1人気'].notna() & df_split['1過去着順'].notna() & df_split['1出走馬数'].notna(), (df_split['1人気'].astype(float) - df_split['1過去着順'].astype(float)) / df_split['1出走馬数'].astype(float))

# 条件の変化
df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
df_split['1場所変化'] = 0
df_split['1場所変化'] = df_split['1場所変化'].mask(df_split['1場所'] == field, 1)
df_split['1フィールド変化'] = 0
df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)

# 不要なカラムを削除
df_split = df_split.drop(['1人気', '1レース名', '1タイム', '1騎手', '1出走馬数', '1馬番', '1斤量', '1馬体重', '1体重増減', '1距離','1騎手', '1馬場', '1フィールド', '1場所', '1過去着順', '1クラス'], axis=1)
df = df.drop(['騎手', '馬場', '馬番', '出走頭数', '性', '斤量'], axis=1)

# 特徴量削減
sou = '1'

# 今走と前走を結合
df_all = pd.concat([df, df_split], axis=1)

######################################################################################
# 以降2~5走の処理
count = 2
sou = '2'
while True:
    # 「2走」から必要なデータにわける
    df_split = df[sou+'走'].str.extract(r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s(\d{2}[.]\d).+(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})', expand=True)
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

    df_split[sou+'着差'] = df_split[sou+'着差'].astype(float) + (df_split[sou+'クラス'].astype(int) * 0.3)

    # 上がり3Fを指数化
    df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 1) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.94 + (df_split[sou+'距離'].astype(float) / 20000)))
    df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 2) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (1.01 + (df_split[sou+'距離'].astype(float) / 20000)))
    df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 3) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.36 + (df_split[sou+'距離'].astype(float) * 1.5 / 100000)))

    # 人気を裏切ったかどうか
    df_split[sou+'人気'] = df_split[sou+'人気'].replace('', np.nan)
    df_split[sou+'過去着順'] = df_split[sou+'過去着順'].replace('', np.nan)
    df_split[sou+'出走馬数'] = df_split[sou+'出走馬数'].replace('', np.nan)
    df_split[sou+'人気差'] = float('nan')
    df_split[sou+'人気差'] = df_split[sou+'人気差'].mask(df_split[sou+'人気'].notna() & df_split[sou+'過去着順'].notna() & df_split[sou+'出走馬数'].notna(), (df_split[sou+'人気'].astype(float) - df_split[sou+'過去着順'].astype(float)) / df_split[sou+'出走馬数'].astype(float))

    # 条件の変化
    df_split[sou+'距離差'] = df['距離'].astype(float) - df_split[sou+'距離'].astype(float)
    df_split[sou+'場所変化'] = 0
    df_split[sou+'場所変化'] = df_split[sou+'場所変化'].mask(df_split[sou+'場所'] == field, 1)
    df_split[sou+'フィールド変化'] = 0
    df_split[sou+'フィールド変化'] = df_split[sou+'フィールド変化'].mask(df_split[sou+'フィールド'] == df['フィールド'], 1)

    # 不要なカラムを削除
    df_split = df_split.drop([sou+'人気', sou+'レース名', sou+'場所', sou+'フィールド', sou+'馬場', sou+'距離', sou+'タイム', sou+'出走馬数', sou+'馬番', sou+'騎手', sou+'斤量', sou+'馬体重', sou+'体重増減', sou+'過去着順', sou+'クラス'], axis=1)

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

# 着順から文字列を排除
indexNames = df_all[(df_all['着順'] != '中止') & (df_all['着順'] != '除外') & (df_all['着順'] != '取消') & (df_all['着順'] != '失格') & (df_all['着順'] != '未定')]
df_all = indexNames

# df_all から Nan を各列の中央値に置換する
df_all = df_all.drop(['2走', '3走', '4走', '5走', 'フィールド', '距離'], axis=1)

for i in range(1,5):
    k = i + 1
    df_all.fillna({f'{i}後3F': df_all[f'{k}後3F'], f'{i}着差': df_all[f'{k}着差'], f'{i}人気差': df_all[f'{k}人気差'], f'{i}距離差': df_all[f'{k}距離差'],
                f'{i}場所変化': df_all[f'{k}場所変化'], f'{i}フィールド変化': df_all[f'{k}フィールド変化'], f'{i}スピード指数': df_all[f'{k}スピード指数']}, inplace=True)

# 目的変数作成
f_ranking = {1: 10, 2: 5, 3: 3, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0,
             '1': 10, '2': 5, '3': 3, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0, '18': 0}

df_all['rank'] = df_all['着順'].map(f_ranking)
df_all = df_all.replace('', '00000')  # ''をエラー検出文字に置換してくれる
df_all = df_all.replace('未定', '00000')
df_all = df_all[~df_all.apply(lambda s: s.str.contains('00000'), axis=1).any(axis=1)]  # エラー検出文字を入れた行以外を抽出
df_all = df_all.astype(float)

# label_gain設定用のリスト
gain_list = [int(i) for i in range(1,15)]

# 1着のみ単勝オッズを保有
df_all['オッズ'] = df_all['単勝オッズ']
df_all['単勝オッズ'] = df_all['単勝オッズ'].where(df_all['着順'].astype(int) == 1, 0)
df_all['単勝オッズ'] = df_all['単勝オッズ'] * 100

# レース内の順序をシャッフル
df_all = df_all.sample(frac=1)
df_all = df_all.sort_values(['レースID'], ascending=True)

# 2023のデータを分離
SplitYear = 202200000000
eval_year = 202100000000
df_test = df_all[df_all['レースID'] >= SplitYear]
df_all = df_all[df_all['レースID'] < SplitYear]
df_eval = df_all[df_all['レースID'] >= eval_year]
df_all = df_all[df_all['レースID'] < eval_year]

# 説明変数,目的変数
X = df_all.drop(['着順', 'rank', '人気', 'オッズ', '単勝オッズ', '馬単'], axis=1)
X2 = df_test.drop(['着順', 'rank', '人気', 'オッズ', '単勝オッズ', '馬単'], axis=1)
y = df_all[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]
y2 = df_test[['rank', 'レースID', 'オッズ', '着順', '単勝オッズ', '馬単']]

# train, eval, testに分割
X_train = X
y_train = y
X_eval = df_eval.drop(['着順', 'rank', '人気', '単勝オッズ', '馬単'], axis=1)
y_eval = df_eval[['rank', 'レースID', '着順', '単勝オッズ', '馬単']]
X_test = X2
y_test = y2

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
y_test_id['オッズ'] = y_test['オッズ']
y_test_id['馬単'] = y_test['馬単'].astype(int)
y_test_id['rank'] = y_test['rank']

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
y_test = y_test.drop(['オッズ'], axis=1)
y_eval = y_eval.drop(['レースID'], axis=1)
y_eval = y_eval.drop(['着順'], axis=1)
y_eval = y_eval.drop(['単勝オッズ'], axis=1)
y_eval = y_eval.drop(['馬単'], axis=1)
y2 = y2.drop(['レースID'], axis=1)
y2 = y2.drop(['着順'], axis=1)
y2 = y2.drop(['単勝オッズ'], axis=1)
y2 = y2.drop(['馬単'], axis=1)

# dataframeを値のみに
X_train = X_train.values
X_test = X_test.values
X_eval = X_eval.values
X2 = X2.values
y_train = y_train.values
y_test = y_test.values
y_eval = y_eval.values
y2 = y2.values

# 学習に使用するデータを設定
lgb_train = lgb.Dataset(X_train, label=y_train, group=train_list)
lgb_eval = lgb.Dataset(X_eval, label=y_eval, reference=lgb_train, group=eval_list)

# パラメータ設定
for seed in range(50):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': 0.01,
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
                                return_cvbooster=True,
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
        'learning_rate': 0.01,
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
        'min_child_samples': best_params['min_child_samples']
    }

    evals_result = {}
    model = lgbm.train(params,
                    lgb_train,  # トレーニングデータの指定
                    valid_names=['train', 'valid'],     # 学習経過で表示する名称
                    valid_sets=[lgb_train, lgb_eval],
                    callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=True),
                                lgbm.record_evaluation(evals_result)]
                    )

    # pklファイルとしてモデルを保存
    with open(f"{tuner_path}{seed}.pickle", "wb") as mk:
        pickle.dump(model, mk)

    # テストデータの予測 (予測クラスを返す)
    y_pred = model.predict(X_test, group=test_list)
    y_test_id[f'result{seed}'] = y_pred

#################### stacking #######################
Z = y_test_id.iloc[:, 6:]
Z['レースID'] = y_test_id['レースID']
w = y_test_id['rank']

# # トレーニングデータ,テストデータの分割
Z_train, Z_test, w_train, w_test = train_test_split(Z, w, test_size=0.20, shuffle=False)

# クエリListを作成
id_count = Z_train['レースID'].value_counts(sort=False)
train2_list = id_count.values.tolist()

id_count = Z_test['レースID'].value_counts(sort=False)
test2_list = id_count.values.tolist()

# レースIDカラムを削除
Z_train = Z_train.drop(['レースID'], axis=1)
Z_test = Z_test.drop(['レースID'], axis=1)

# カラム名削除
Z_train = Z_train.values
Z_test = Z_test.values
w_train = w_train.values
w_test = w_test.values

# 学習に使用するデータを設定
zlgb_train = lgb.Dataset(Z_train, label=w_train, group=train2_list)
zlgb_test = lgb.Dataset(Z_test, label=w_test, reference=zlgb_train, group=test2_list)

for seed in range(50):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,2,3],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': 0.01,
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
        'learning_rate': 0.01,
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
        'min_child_samples': best_params['min_child_samples']
    }

    evals_result = {}
    model_s = lgbm.train(params,
                    zlgb_train,  # トレーニングデータの指定
                    valid_names=['train', 'valid'],     # 学習経過で表示する名称
                    valid_sets=[zlgb_train, zlgb_test],
                    callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=True),
                                lgbm.record_evaluation(evals_result)]
                    )
    
    # pklファイルとしてモデルを保存
    # if seed == 1 or seed == 8:
    with open(f"{tuner_path}{seed}-stack.pickle", "wb") as mk:
        pickle.dump(model_s, mk)

#####################################################

# seedの違う50モデルの平均を算出(stacking前のモデル)
y_test_id['Average'] = y_test_id.iloc[:, 6:].mean(axis=1)
y_test_id = y_test_id.sort_values(['レースID', 'Average'], ascending=[False, False])

# 予測スコアを標準化
try:
    counts = 0
    for k in test_list:
        mean_df = y_test_id.iloc[counts:counts+k, -1].mean()
        std_df = y_test_id.iloc[counts:counts+k, -1].std()
        y_test_id.iloc[counts:counts+k, -1] = (y_test_id.iloc[counts:counts+k, -1] - mean_df) / std_df
        counts += k
except:
    pass

# 回収率計算
plotlist = []
holdlist = []
countlist = []
plotlist2 = []
countlist2 = []
# 単勝
for k in range(0, 21):
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
            count += 1
        # 1着予想馬のスコアのみで判定
        if ((y_test_id.iloc[num, -1].astype(float))) >= threshold:
            sum_ += (y_test_id.iloc[num, 2]).astype(float)
            count_ += 1
        num += i
    countlist.append(count)
    plotlist.append(sum / (count * 100))
    holdlist.append(threshold)
    countlist2.append(count_)
    plotlist2.append(sum_ / (count_ * 100))
print(f"各閾値ごとの買い目点数(スコア差)\n{countlist}")
print(f"各閾値ごとの回収率(スコア差)\n{plotlist}")
plt.title('単勝(スコア差)')
plt.plot(holdlist, plotlist)
plt.show()
print(f"各閾値ごとの買い目点数(スコアのみ)\n{countlist2}")
print(f"各閾値ごとの回収率(スコアのみ)\n{plotlist2}")
plt.title('単勝(スコアのみ)')
plt.plot(holdlist, plotlist2)
plt.show()