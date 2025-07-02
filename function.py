import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pickle
from IPython.display import display
import numpy as np
import warnings
import lightgbm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

warnings.simplefilter('ignore')

# ID:202405020208
def chukyo(race_id, odds):
    # 会場
    field = 'tyukyo'

    field_num = 9

    # ファイル数
    file_num = 5

    # ファイルパス
    version = 'test70'
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}{version}_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    # options.add_argument("--headless")
    options.add_argument("--headless")
    options.add_argument('--log-level=3')

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']

    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)
    # display(df_shutuba.columns)
    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')
    # df['血統'] = pd.to_numeric((df['父馬'] * 10).astype(str) + df['母父馬'].astype(str), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬'], axis=1)

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

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', 
                             '1馬場', '1出走馬数', '1馬番', '1人気', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', 
                             '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', '1ペース差', '2過去着順', '2距離', '2人気', '2後3F', '2着差', '2スピード指数', 
                             '2クラス', '2クラス差', '3過去着順', '3距離', '3人気', '3後3F', '3着差', '3スピード指数', '3クラス', '3クラス差', '4過去着順', '4距離', '4人気', '4後3F', '4着差', 
                             '4スピード指数', '4クラス', '4クラス差', '5過去着順', '5距離', '5人気', '5後3F', '5着差', '5スピード指数', '5クラス', '5クラス差', 'best着差', 'bestスピード指数', 
                             'av着差', 'avスピード指数', '上昇度'], axis=1)
    # print(df_all.columns)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.4:
        return results.iloc[0]['馬番']
    
    else:
        return None


def nakayama(race_id, odds):
    # 会場
    field = 'nakayama'

    field_num = 1

    # ファイル数
    file_num = 5

    # ファイルパス
    version = 'test70'
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}{version}_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()

    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    options.add_argument("--headless")
    options.add_argument('--log-level=3')

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']

    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1)
    # display(df)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')
    # df['血統'] = pd.to_numeric((df['父馬'] * 10).astype(str) + df['母父馬'].astype(str), errors='coerce')

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

    df_split['人気増減'] = df_split['1人気'].astype(float) - df['人気'].astype(float)
    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名',sou+'騎手'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/1class_dict{field_num}.pkl', "rb") as dd:
        classsa_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '騎手', '母父馬'], axis=1)

    list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気',  '斤量', '後3F', '馬体重', '体重増減', '着差', 'スピード指数']

    for i in range(1,5):
        k = i + 1
        for v in list_columns:
            df_all.fillna({f'{i}{v}': df_all[f'{k}{v}']}, inplace=True)
            df_all.fillna({f'{k}{v}': df_all[f'{i}{v}']}, inplace=True)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', 
    '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '人気増減', '1距離差', '1場所変化', '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', 
    '1ペース差', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', 
    '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', 
    '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', 
    '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 
    'best着差', 'bestスピード指数', 'av着差', 'avスピード指数'], axis=1)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.4:
        return results.iloc[0]['馬番']
    
    else:
        return None


def tokyo(race_id, odds):
    # 会場
    field = 'tokyo'

    field_num = 2

    # ファイル数
    file_num = 5

    # ファイルパス
    version = 'test70'
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}{version}_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    options.add_argument("--headless")
    options.add_argument('--log-level=3')
    options.add_argument("--disable-blink-features=AutomationControlled")

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']
    
    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')
    # df['血統'] = pd.to_numeric((df['父馬'] * 10).astype(str) + df['母父馬'].astype(str), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split[sou+'着差'] = df_split[sou+'着差'].astype(float) + (df_split[sou+'クラス'].astype(int) * 0.5)

        # 上がり3Fを指数化
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 1) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.94 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 2) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (1.01 + (df_split[sou+'距離'].astype(float) / 20000)))
        df_split[sou+'後3F'] = df_split[sou+'後3F'].mask((df_split[sou+'フィールド'] == 3) & df_split[sou+'後3F'].notna() & df_split[sou+'距離'].notna(), df_split[sou+'後3F'].astype(float) / (0.36 + (df_split[sou+'距離'].astype(float) * 1.5 / 100000)))

        df_split = df_split.drop([sou+'レース名',sou+'騎手'], axis=1)

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


    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬', '1コーナー通過順'], axis=1)

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

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', 
    '1斤量', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', '1ペース差', '2場所', '2過去着順', 
    '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', '3場所', '3過去着順', '3フィールド', '3距離', 
    '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', 
    '4出走馬数', '4馬番', '4人気', '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', 
    '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度'], axis=1)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.6:
        return results.iloc[0]['馬番']
    
    else:
        return None
    

def kyoto(race_id, odds):
    # 会場
    field = 'kyoto'

    field_num = 3

    # ファイル数
    file_num = 5

    # ファイルパス
    version = 'test70'
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}{version}_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    # options.add_argument("--headless")
    options.add_argument("--headless") 
    options.add_argument('--log-level=3')
    options.add_argument("--disable-blink-features=AutomationControlled")

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']
        
    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名',sou+'騎手'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬'], axis=1)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', 
                             '1出走馬数', '1馬番', '1人気', '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', 
                             '平均クラス', '平均ペース', '1クラス差', '1ペース差', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', 
                             '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', 
                             '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', 
                             '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', 
                             '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 'best着差', 'bestスピード指数', 'av着差', 'avスピード指数'], axis=1)

    # df_all = df_all.drop(['オッズ'], axis=1)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 2.0 and results.iloc[0]['ex'] < 4.2:
        return results.iloc[0]['馬番']
    
    else:
        return None
    

def nigata(race_id, odds):
    # 会場
    field = 'nigata'

    field_num = 8

    # ファイル数
    file_num = 5

    # ファイルパス
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}test20_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    options.add_argument("--headless")
    options.add_argument('--log-level=3')

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']
        
    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', 'メモ', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名',sou+'騎手'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬'], axis=1)

    list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気',  '斤量', '後3F', '馬体重', '体重増減', '着差', 'スピード指数']

    for i in range(1,5):
        k = i + 1
        for v in list_columns:
            df_all.fillna({f'{i}{v}': df_all[f'{k}{v}']}, inplace=True)
            df_all.fillna({f'{k}{v}': df_all[f'{i}{v}']}, inplace=True)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', 
    '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', 
    '1ペース差', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', 
    '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', 
    '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', 
    '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 
    'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', 'オッズ'], axis=1)

    df_all = df_all.drop(['オッズ'], axis=1)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.3:
        return results.iloc[0]['馬番']
    
    else:
        return None


def hukushima(race_id, odds):
    # 会場
    field = 'hukushima'

    field_num = 7

    # ファイル数
    file_num = 5

    # ファイルパス
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}test20_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    options.add_argument("--headless")
    options.add_argument('--log-level=3')
    options.add_argument("--disable-blink-features=AutomationControlled")

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']
        
    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)

    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', 'メモ', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名',sou+'騎手'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬'], axis=1)

    list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気',  '斤量', '後3F', '馬体重', '体重増減', '着差', 'スピード指数']

    for i in range(1,5):
        k = i + 1
        for v in list_columns:
            df_all.fillna({f'{i}{v}': df_all[f'{k}{v}']}, inplace=True)
            df_all.fillna({f'{k}{v}': df_all[f'{i}{v}']}, inplace=True)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', 
    '1斤量', '1コーナー通過順', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', 
    '1ペース差', '2場所', '2過去着順', '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', 
    '3場所', '3過去着順', '3フィールド', '3距離', '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', 
    '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', '4出走馬数', '4馬番', '4人気', '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', 
    '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 
    'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', 'オッズ'], axis=1)

    df_all = df_all.drop(['オッズ'], axis=1)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.1:
        return results.iloc[0]['馬番']
    
    else:
        return None


def kokura(race_id, odds):
    # 会場
    field = 'kokura'

    field_num = 10

    # ファイル数
    file_num = 5

    # ファイルパス
    version = 'test70'
    horse_path = "./pickle-dict/horse_jra.pkl"
    femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
    jockey_path = "./pickle-dict/jockey_jra.pkl"
    tuner_path = f"./pickle-tuner/{field}{version}_"

    # 行を全表示（行の数）
    pd.set_option("display.max_rows", None)

    # 列を全表示（列の数）
    pd.set_option("display.max_columns", None)

    # データフレームを生成
    df_shutuba = pd.DataFrame()

    # 該当ページ(レース)をスクレイピング
    url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]

    # マルチカラムを解除
    df_now.columns = df_now.columns.droplevel()
    # print(df_now.columns, df_past.columns)
    df_result_past = pd.merge(df_now, df_past, on='馬番')
    r = requests.get(url_race, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = re.findall(r'\d+', data1)[2]
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]

    df_result_past['レースID'] = race_id

    # ブラウザのオプションを格納する変数をもらってきます。
    options = Options()

    # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    # options.add_argument("--headless")
    options.add_argument("--headless")
    options.add_argument('--log-level=3')

    # ブラウザを起動する
    driver = webdriver.Chrome(options=options)

    # ブラウザでアクセスする
    driver.get(url_race)

    # HTMLを文字コードをUTF-8に変換してから取得します。
    html = driver.page_source.encode('utf-8')

    # BeautifulSoupで扱えるようにパースします
    soup = BeautifulSoup(html, "html.parser")

    # tableを取得(js反映)
    el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # display(df_shutuba.columns)

    # # インデックス番号振り直し、オッズを格納
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()
    # try:
    #     df_shutuba['オッズ'] = df1['オッズ 更新']
    # except KeyError:
    #     df_shutuba['オッズ'] = df1['オッズ']

    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

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

    # カラム作成
    df_shutuba['父馬'] = df_shutuba['馬名_y'].str.extract(r'(\w+\s)', expand=True)
    df_shutuba['間隔'] = df_shutuba['馬名_y'].str.extract(r'(\d+)', expand=True)
    df_shutuba['母父馬'] = df_shutuba['馬名_y'].str.extract(r'(\(\D+\))', expand=True)

    horse_mapping = {}
    with open(horse_path, mode="rb") as f:
        horse_mapping = pickle.load(f)
    with open(femal_horse_path, mode="rb") as f:
        femal_horse_mapping = pickle.load(f)
    # df_shutuba['父馬'] = df_shutuba['父馬'].map(horse_mapping)
    # df_shutuba['母父馬'] = df_shutuba['母父馬'].map(femal_horse_mapping)
    # display(df_shutuba.columns)
    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1)
    # display(df.columns)

    # 取り消し馬を削除
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')
    df['母父馬'] = pd.to_numeric(df['母父馬'].map(femal_horse_mapping), errors='coerce')
    # df['血統'] = pd.to_numeric((df['父馬'] * 10).astype(str) + df['母父馬'].astype(str), errors='coerce')

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

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']

    df_split = df_split.drop(['1レース名', '1騎手'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # クエリListを作成
    df_all['平均クラス'] = np.nan
    df_all['平均ペース'] = np.nan
    df_all['平均クラス'] = df_all['1クラス'].mean().astype(int)
    df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')
    df_all['1ペース差'] = df_all['平均ペース'] - df_all['1コーナー通過順']


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

        df_split = df_split.drop([sou+'レース名', sou+'騎手',sou+'場所',sou+'フィールド',sou+'馬場',sou+'タイム',sou+'出走馬数', sou+'馬番',sou+'馬体重', sou+'体重増減',sou+'斤量'], axis=1)

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

    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')

    # best, av
    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # いらないカラムを削除
    df_all = df_all.drop(['2走', '3走', '4走', '5走', '母父馬', '1コーナー通過順'], axis=1)

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

    # 空白削除
    for i in df_all.columns:
        df_all[i] = df_all[i].replace('', np.nan)

    # result作成
    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    # 説明変数をdataXに格納
    # カラム順を整列
    df_all = df_all.reindex(['馬番', '斤量', '騎手', '人気', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '性', '齢', '1場所', '1過去着順', '1フィールド', '1距離', '1タイム', '1馬場', '1出走馬数', '1馬番', '1人気', 
                            '1斤量', '1後3F', '1馬体重', '1体重増減', '1着差', '1クラス', '1スピード指数', '1距離差', '1場所変化', '1フィールド変化', '平均クラス', '平均ペース', '1クラス差', '1ペース差', '2場所', '2過去着順', 
                            '2フィールド', '2距離', '2タイム', '2馬場', '2出走馬数', '2馬番', '2人気', '2斤量', '2後3F', '2馬体重', '2体重増減', '2着差', '2スピード指数', '2クラス', '3場所', '3過去着順', '3フィールド', '3距離', 
                            '3タイム', '3馬場', '3出走馬数', '3馬番', '3人気', '3斤量', '3後3F', '3馬体重', '3体重増減', '3着差', '3スピード指数', '3クラス', '4場所', '4過去着順', '4フィールド', '4距離', '4タイム', '4馬場', 
                            '4出走馬数', '4馬番', '4人気', '4斤量', '4後3F', '4馬体重', '4体重増減', '4着差', '4スピード指数', '4クラス', '5場所', '5過去着順', '5フィールド', '5距離', '5タイム', '5馬場', '5出走馬数', '5馬番', 
                            '5人気', '5斤量', '5後3F', '5馬体重', '5体重増減', '5着差', '5スピード指数', '5クラス', 'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度'], axis=1)
    # print(df_all.columns)

    dataX = df_all.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(dataX)
        result[f'result{i}'] = y_pred

    Z = result.iloc[:, 2:]
    Z['Average'] = Z.iloc[:, :].mean(axis=1)
    Z['馬番'] = result['馬番']
    Z = Z.sort_values(['Average'], ascending=[False])

    for k in range(1, 6):
        mean_df = Z[f'result{k}'].mean()
        std_df = Z[f'result{k}'].std()
        Z[f'result{k}'] = (Z[f'result{k}'] - mean_df) / std_df

    Z['Average'] = Z.iloc[:, Z.columns.get_loc(f'result1'):Z.columns.get_loc(f'result5')+1].mean(axis=1)
    Z = Z.sort_values(['Average'], ascending=[False])

    Z['lower_diff'] = Z['Average'] - Z['Average'].shift(-1)
    Z['upper_diff'] = Z['Average'] - Z['Average'].shift(1)

    Z.iat[-1, -2] = float('nan')
    Z.iat[0, -1] = float('nan')

    # レース内の順序をシャッフル
    Z = Z.sample(frac=1)
    results = pd.DataFrame()
    results['馬番'] = Z['馬番']
    Z = Z.drop(['馬番'], axis=1)

    # カラム名削除
    Z = Z.values

    for i in range(1, file_num+1):
        # モデル呼び出し
        with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
            model = pickle.load(f)

        # テストデータの予測 ((各クラスの予測確率 [クラス0の予測確率,クラス1の予測確率,クラス2の予測確率] を返す))
        y_pred = model.predict(Z)
        results[f'result{i}'] = y_pred

    result = result.sort_values(['馬番'], ascending=[True])

    results['Average'] = results.iloc[:, 1:].mean(axis=1)

    # 標準化
    mean_df = results['Average'].mean()
    std_df = results['Average'].std()

    with open(f'./pickle-dict/pre_dict{field_num}_{version}.pkl', "rb") as f:
        pre_d = pickle.load(f)

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    # display(results)
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['score'] = results['score'].map('{:.1f}'.format)
    results['ex'] = results['score'].astype(float).map(pre_d)
    # display(results)
    results['ex'] = results['ex'].astype(float) * results['オッズ'].astype(float)

    results = results[['馬番', 'score', 'オッズ', 'ex']]

    display(results)

    if results.iloc[0]['ex'] >= 1.5:
        return results.iloc[0]['馬番']
    
    else:
        return None

# print(tokyo(202507010301, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]))