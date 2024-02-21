import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pickle
from IPython.display import display
import numpy as np
import warnings

warnings.simplefilter('ignore')

# カラム順
# ['馬番' '斤量' '騎手' '距離' 'フィールド' '馬場' '出走頭数' 'レースID' '父馬' '間隔' '性' '齢' '体重増減'
#  '1場所' '1過去着順' '1フィールド' '1距離' '1馬場' '1後3F' '1着差' '1クラス' '1人気差' '2過去着順'       
#  '2後3F' '2着差' '2クラス' '2人気差' '3過去着順' '3後3F' '3着差' '3クラス' '3人気差' '4過去着順'
#  '4後3F' '4着差' '4クラス' '4人気差' '5過去着順' '5後3F' '5着差' '5クラス' '5人気差']

# 会場
field = 'tokyo'

# ファイルパス
horse_path = "./pickle-dict/horse_jra.pkl"
femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
jockey_path = "./pickle-dict/jockey_jra.pkl"
tuner_path = f"./pickle-tuner/{field}2_"

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)

# 列を全表示（列の数）
pd.set_option("display.max_columns", None)

# データフレームを生成
df_shutuba = pd.DataFrame()

# レースIDを入力
while True:
    race_id = input('レースID:')
    if race_id:
        break
    print('レースIDを入力してください。')

# 該当ページ(レース)をスクレイピング
url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
print(url_race)
df_now = pd.read_html(url_race)[0]
df_past = pd.read_html(url_past)[0]

# マルチカラムを解除
df_now.columns = df_now.columns.droplevel()

df_result_past = pd.merge(df_now, df_past, on='馬番')
r = requests.get(url_race)
soup = BeautifulSoup(r.content, 'html.parser')
data1 = soup.find('div', class_='RaceData01').text
data2 = soup.find('div', class_='RaceData02').text
df_result_past['距離'] = re.findall(r'\d+', data1)[2]
df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

df_result_past['レースID'] = race_id
df_shutuba = pd.concat([df_shutuba, df_result_past])

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

# カラム作成
df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)


horse_mapping = {}
with open(horse_path, mode="rb") as f:
    horse_mapping = pickle.load(f)
with open(femal_horse_path, mode="rb") as f:
    femal_horse_mapping = pickle.load(f)
df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

# いらないカラムを消す
df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', 'メモ', '人気'], axis=1)

# カラム順を整列
df = df.reindex(['馬番', '斤量', '騎手', '前走', '2走', '3走', '4走', '5走', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '母父馬', '間隔', '性齢', '馬体重(増減)', 'オッズ'], axis=1)

# 取り消し馬を削除
indexNames = df[df['オッズ'] != '--']
df = indexNames

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
df_split['1コーナー通過順'] = df_split['1コーナー通過順'].str[-4:-1].astype(float)

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

df_split['1着差'] = df_split['1着差'].astype(float) + (df_split['1クラス'].astype(int) * 0.3)

# 騎手のユニーク値から辞書をつくる
jockey_mapping = {}
with open(jockey_path, mode="rb") as f:
    jockey_mapping = pickle.load(f)

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

# 不要なカラムを削除
df_split['距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
df_split['1場所変化'] = 0
df_split['1場所変化'] = df_split['1場所変化'].mask(df_split['1場所'] == 1, 1)
df_split['1フィールド変化'] = 0
df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)

df_split = df_split.drop(['1人気', '1レース名', '1タイム', '1騎手', '1出走馬数', '1馬番', '1斤量', '1馬体重', '1体重増減', '1距離','1騎手', '1馬場', '1フィールド', '1場所', '1過去着順', '1クラス'], axis=1)
df = df.drop(['騎手', '出走頭数', '性', '斤量', '馬場'], axis=1)

# 今走と前走を結合
df_all = pd.concat([df, df_split], axis=1)

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
    df_split[sou+'場所変化'] = df_split[sou+'場所変化'].mask(df_split[sou+'場所'] == 1, 1)
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

result = pd.DataFrame()
result['馬番'] = df_all['馬番']
result['オッズ'] = df_all['オッズ']

# df_all から Nan を各列の中央値に置換、削除する
df_all = df_all.drop(['オッズ', '2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離'], axis=1)

for i in range(1,5):
    k = i + 1
    df_all.fillna({f'{i}後3F': df_all[f'{k}後3F'], f'{i}着差': df_all[f'{k}着差'], f'{i}人気差': df_all[f'{k}人気差'], f'{i}距離差': df_all[f'{k}距離差'],
                f'{i}場所変化': df_all[f'{k}場所変化'], f'{i}フィールド変化': df_all[f'{k}フィールド変化'], f'{i}スピード指数': df_all[f'{k}スピード指数']}, inplace=True)

# 説明変数をdataXに格納
dataX = df_all.values

for i in range(50):
    # モデル呼び出し
    with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
        model = pickle.load(f)

    # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
    y_pred = model.predict(dataX)
    result[f'result{i}'] = y_pred

Z = result.iloc[:, 2:].values
for i in range(50):
    with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)
    y_pred = model.predict(Z)

    result[f'stack{i}'] = y_pred

# stack average
for i in range(50):
    mean_df = result[f'stack{i}'].mean()
    std_df = result[f'stack{i}'].std()
    result[f'score{i}'] = (result[f'stack{i}'] - mean_df) / std_df
result['s-Average'] = result.iloc[:, 102:].mean(axis=1)
mean_df = result["s-Average"].mean()
std_df = result['s-Average'].std()
result['score'] = (result['s-Average'] - mean_df) / std_df

# normal average
for i in range(50):
    mean_df = result[f'result{i}'].mean()
    std_df = result[f'result{i}'].std()
    result[f'nscore{i}'] = (result[f'result{i}'] - mean_df) / std_df
result['Average'] = result.iloc[:, 153:].mean(axis=1)
mean_df = result["Average"].mean()
std_df = result['Average'].std()
result['nscore'] = (result['Average'] - mean_df) / std_df

result = result.sort_values('s-Average', ascending=False)
results = result[['馬番', 'オッズ', 'score']]
result = result.sort_values('Average', ascending=False)
result = result[['馬番', 'オッズ', 'nscore']]
display(result)
display(results)
