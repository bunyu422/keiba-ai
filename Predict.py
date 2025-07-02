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

# 1.7: 差が0.4未満(0.3-0.4 good) 103%
# 1.1-1.6: 0.6未満(0.2-0.3 good) 115-125%
# 1: 差が1未満(0.4-0.5 good) 157%

# カラム順
# ['馬番' '斤量' '騎手' '距離' 'フィールド' '馬場' '出走頭数' 'レースID' '父馬' '間隔' '性' '齢' '体重増減'
#  '1場所' '1過去着順' '1フィールド' '1距離' '1馬場' '1後3F' '1着差' '1クラス' '1人気差' '2過去着順'       
#  '2後3F' '2着差' '2クラス' '2人気差' '3過去着順' '3後3F' '3着差' '3クラス' '3人気差' '4過去着順'
#  '4後3F' '4着差' '4クラス' '4人気差' '5過去着順' '5後3F' '5着差' '5クラス' '5人気差']

# {'中山': 1, '東京': 2, '京都': 3, '阪神': 4, '札幌': 5, '函館': 6, '福島': 7, '新潟': 8, '中京': 9, '小倉': 10,
#  '帯広': 11, '門別': 12, '盛岡': 13, '水沢': 14, '浦和': 15, '船橋': 16, '大井': 17, '川崎': 18, '金沢': 19, '笠松': 20,
#  '名古屋': 21, '園田': 22, '姫路': 23, '高知': 24, '佐賀': 25}

# スコア差毎の勝率
# 0: 0.27979274611398963, 
# 1: 0.2864864864864865, 
# 2: 0.29545454545454547, 
# 3: 0.2926829268292683, 
# 4: 0.2866666666666667, 
# 5: 0.3252032520325203, 
# 6: 0.3557692307692308, 
# 7: 0.3516483516483517, 
# 8: 0.3424657534246575, 
# 9: 0.3103448275862069, 
# 10: 0.3333333333333333, 
# 11: 0.34285714285714286, 
# 12: 0.39285714285714285, 
# 13: 0.45, 
# 14: 0.5, 
# 15: 0.5714285714285714, 
# 16: 0.6, 
# 17: 1.0

'''
0.2757352941176471, 0.27608695652173915, 0.2734375, 0.2779605263157895, 0.2903225806451613, 0.31295843520782396, 0.33045977011494254, 0.3356401384083045, 0.34334763948497854, 
0.38333333333333336, 0.3602941176470588, 0.35185185185185186, 0.3764705882352941, 0.4126984126984127, 0.3958333333333333, 0.4, 0.5, 0.5, 0.4, 0.5, 0.5, 0.3333333333333333, 
0.5, 1.0, 1.0, 1.0, 1.0, 1.0
'''

# 東京 mean：-0.07393992750213065, std：0.1178771705391725
# 東京3 -0.0861163211048716 0.13771350740836602
# 東京4 -0.07703596810396432 0.12303472881495063
# 東京test -0.08483566537673962 0.12649441672624553
# 東京test4 stack: -0.028265160141442355 0.06595920023456842
# 東京test6: -0.04041285879814497 0.08502220294881775
# stack：  -0.024201518832282286 0.0405755294488378
# 東京test.stack：-0.034462296358653 0.05681036414750313 
# 東京5 -0.045617794913679836 0.11121248701713331
# 阪神 mean：-0.039798236081803766, std：0.06909357874563701
# 阪神2 -0.054095384894612, 0.09237264457442314
# 阪神3 -0.06475473513151409 0.1076269776383134
# 中山3 -0.06520717673554564 0.10731898340933799
# 福島3 -0.04390945894809122 0.06391078042626286

# 会場
#field = 'nakayama'
#field = 'hanshin'
# field = 'tokyo'
#field = 'hukushima'
field = 'hakodate'

# field_num = 2
field_num = 6

# 平均, 偏差
# 中山
# mean_df = -0.06520717673554564
# std_df = 0.10731898340933799

# 阪神
# mean_df = -0.06475473513151409
# std_df = 0.1076269776383134

# 東京
# mean_df = -0.02858950324393746
# std_df = 0.04591514602541163

# 福島
# mean_df = -0.04390945894809122
# std_df = 0.06391078042626286

# 函館
mean_df = -0.11248551456866171
std_df = 0.09597281646494404

# ファイル数
file_num = 5

# ファイルパス
horse_path = "./pickle-dict/horse_jra.pkl"
femal_horse_path = "./pickle-dict/femal_horse_jra.pkl"
jockey_path = "./pickle-dict/jockey_jra.pkl"
tuner_path = f"./pickle-tuner/{field}test10_"

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)

# 列を全表示（列の数）
pd.set_option("display.max_columns", None)

# レースIDを入力
while True:
    race_id = input('race ID:')
    if race_id:
        break
    print('please input race ID')

for t in range(1, 13):
    # データフレームを生成
    df_shutuba = pd.DataFrame()
    
    if int(t) < 10:
        race_id = race_id[:10] + '0' + str(t)
    else:
        race_id = race_id[:10] + str(t)

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

    # # ブラウザのオプションを格納する変数をもらってきます。
    # options = Options()

    # # Headlessモードを有効にする（コメントアウトするとブラウザが実際に立ち上がります）
    # options.add_argument("--headless")
    # options.add_argument('--log-level=3')

    # # ブラウザを起動する
    # driver = webdriver.Chrome(options=options)

    # # ブラウザでアクセスする
    # driver.get(url_race)

    # # HTMLを文字コードをUTF-8に変換してから取得します。
    # html = driver.page_source.encode('utf-8')

    # # BeautifulSoupで扱えるようにパースします
    # soup = BeautifulSoup(html, "html.parser")

    # # tableを取得(js反映)
    # el=driver.find_element(By.CLASS_NAME, "RaceTableArea") #classでテーブルを指定
    # html=el.get_attribute("outerHTML")#table要素を含むhtmlを取得
    # df1=pd.read_html(html)[0]#tableをDataFrameに格納

    # df結合
    df_shutuba = pd.concat([df_shutuba, df_result_past])

    # # インデックス番号振り直し、オッズを格納
    # df1.reset_index(inplace=True, drop=True)
    # df_shutuba['オッズ'] = df1['オッズ']

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
    
    '''pro_dict = {'-0.0': 0.02617801047120419,
                '-0.1': 0.03655352480417755,
                '-0.2': 0.04132231404958678,
                '-0.3': 0.04054054054054054,
                '-0.4': 0.0225,
                '-0.5': 0.027833001988071572,
                '-0.6': 0.02111324376199616 ,
                '-0.7': 0.011135857461024499,
                '-0.8': 0.02857142857142857,
                '-0.9': 0.010101010101010102,
                '-1.0': 0.014423076923076924,
                '-1.1': 0.01858736059479554,
                '-1.2': 0.01312910284463895,
                '-1.3': 0.00554016620498615 ,
                '-1.4': 0.008130081300813009,
                '-1.5': 0.0,
                '-1.6': 0.0,
                '-1.7': 0.0,
                '-1.8': 0.0,
                '0.0': 0.03614457831325301,
                '0.1': 0.060240963855421686,
                '0.2': 0.05869074492099323,
                '0.3': 0.04751131221719457,
                '0.4': 0.0759493670886076,
                '0.5': 0.09868421052631579,
                '0.6': 0.10862619808306709,
                '0.7': 0.08615384615384615,
                '0.8': 0.13291139240506328,
                '0.9': 0.13550135501355012,
                '1.0': 0.07142857142857142,
                '1.1': 0.15654952076677317,
                '1.2': 0.19424460431654678,
                '1.3': 0.14285714285714285,
                '1.4': 0.1391304347826087,
                '1.5': 0.2009132420091324,
                '1.6': 0.1781609195402299,
                '1.7': 0.208955223880597,
                '1.8': 0.20958083832335328,
                '1.9': 0.34810126582278483,
                '2.0': 0.32061068702290074,
                '2.1': 0.29523809523809524,
                '2.2': 0.5853658536585366}'''

    pro_dict = {'0.8': 0.0,
                '0.9': 0.0,
                '1.0': 0.0,
                '1.1': 0.0,
                '1.2': 0.23076923076923078,
                '1.3': 0.21621621621621623,
                '1.4': 0.20588235294117646,
                '1.5': 0.5151515151515151}

    pro1_dict = {'0.2': 0.0,
                '0.3': 0.2,
                '0.4': 0.5,
                '0.5': 0.5,
                '0.6': 0.4,
                '0.7': 1.0,
                '0.8': 0.25,
                '0.9': 0.16666666666666666,
                '1.0': 0.125,
                '1.1': 0.25,
                '1.2': 0.4444444444444444,
                '1.3': 0.2857142857142857,
                '1.4': 0.22580645161290322,
                '1.5': 0.2222222222222222,
                '1.6': 0.225,
                '1.7': 0.22727272727272727,
                '1.8': 0.2127659574468085,
                '1.9': 0.38461538461538464,
                '2.0': 0.3203125,
                '2.1': 0.29523809523809524,
                '2.2': 0.5853658536585366,}

    pro3_dict = {'-0.0': 0.0,
                '-0.1': 0.0,
                '-0.2': 0.0,
                '-0.3': 0.0,
                '-0.4': 0.0,
                '0.0': 0.0,
                '0.1': 0.16666666666666666,
                '0.2': 0.08108108108108109,
                '0.3': 0.0975609756097561,
                '0.4': 0.1875,
                '0.5': 0.14035087719298245,
                '0.6': 0.14084507042253522,
                '0.7': 0.15584415584415584,
                '0.8': 0.19047619047619047,
                '0.9': 0.18571428571428572,
                '1.0': 0.09815950920245399,
                '1.1': 0.19791666666666666,
                '1.2': 0.22631578947368422,
                '1.3': 0.17142857142857143,
                '1.4': 0.15151515151515152,
                '1.5': 0.20588235294117646,
                '1.6': 0.18674698795180722,
                '1.7': 0.2076923076923077,
                '1.8': 0.2073170731707317,
                '1.9': 0.34810126582278483,
                '2.0': 0.3230769230769231,
                '2.1': 0.29523809523809524,
                '2.2': 0.5853658536585366}

    pro1_dict = {0.7: 1.0,
                0.9: 0.3333333333333333,
                1.0: 1.0,
                1.1: 0.5,
                1.2: 0.0,
                1.3: 0.5,
                1.4: 0.14285714285714285,
                1.5: 0.36363636363636365,
                1.6: 0.23529411764705882,
                1.7: 0.13333333333333333,
                1.8: 0.2777777777777778,
                1.9: 0.0625,
                2.0: 0.27586206896551724,
                2.1: 0.16666666666666666,
                2.2: 0.6666666666666666,
                2.3: 0.3448275862068966}


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

    df_shutuba['父馬'] = pd.to_numeric(df_shutuba['父馬'].map(horse_mapping), errors='coerce')
    df_shutuba['母父馬'] = pd.to_numeric(df_shutuba['母父馬'].map(femal_horse_mapping), errors='coerce')
    # df_shutuba['血統'] = pd.to_numeric((df_shutuba['父馬'] * 10).astype(str) + df_shutuba['母父馬'].astype(str), errors='coerce')


    # いらないカラムを消す
    df = df_shutuba.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', 'レースID', '登録', 'メモ', '人気'], axis=1)

    # カラム順を整列
    df = df.reindex(['馬番', '性齢', '斤量', '騎手', '前走', '2走', '3走', '4走', '5走', '距離', 'フィールド', '馬場', '出走頭数', '父馬', '間隔', '母父馬',  '馬体重(増減)', 'オッズ'], axis=1)

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
    df_split['1コーナー通過順'] = df_split['1コーナー通過順'].str[-4:-1].apply(lambda x : x if x != ' ' else None)
    df_split['1コーナー通過順'] = df_split['1コーナー通過順'].astype(float)

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
    # df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    # df_split['1場所変化'] = 0
    # df_split['1場所変化'] = df_split['1場所変化'].mask(df_split['1場所'] == field_num, 1)
    # df_split['1フィールド変化'] = 0
    # df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)

    # df_split['1フィールド変化*スピード指数'] = df_split['1フィールド変化'] * df_split['1スピード指数']
    # df_split['1場所変化*スピード指数'] = df_split['1場所変化'] * df_split['1スピード指数']
    # df_split['1距離差*スピード指数'] = df_split['1距離差'] * df_split['1スピード指数']
    # df_split['1コーナー*3F'] = df_split['1コーナー通過順'] * df_split['1後3F']
    # df_split['馬番差'] = (df['出走頭数'].astype(float) / df['馬番'].astype(float)) - (df_split['1出走馬数'].astype(float) / df_split['1馬番'].astype(float))
    # df_split['騎手変化'] = 1
    # df_split['騎手変化'] = df_split['騎手変化'].mask(df_split['1騎手'] == df['騎手'], 0)
    # df_split['騎手変化*スピード指数'] = df_split['騎手変化'] * df_split['1スピード指数']

    df_split['1距離差'] = df['距離'].astype(float) - df_split['1距離'].astype(float)
    df_split['1場所変化'] = df_split['1場所'] - field_num
    # df_split['1フィールド変化'] = 0
    # df_split['1フィールド変化'] = df_split['1フィールド変化'].mask(df_split['1フィールド'] == df['フィールド'], 1)
    df_split['1フィールド変化'] = df_split['1フィールド'] - df['フィールド']
    df_split['馬番差'] = (df['出走頭数'].astype(float) / df['馬番'].astype(float)) - (df_split['1出走馬数'].astype(float) / df_split['1馬番'].astype(float))
    df_split['騎手変化'] = 1
    df_split['騎手変化'] = df_split['騎手変化'].mask(df_split['1騎手'] == df['騎手'], 0)

    df_split = df_split.drop(['1レース名'], axis=1)
    # df_split = df_split.drop(['1人気', '1レース名', '1タイム', '1騎手', '1出走馬数', '1馬番', '1斤量', '1馬体重', '1体重増減', '1距離','1騎手', '1馬場', '1フィールド', '1場所', '1クラス'], axis=1)
    # df = df.drop(['騎手', '出走頭数', '性', '斤量', '馬場'], axis=1)

    # 今走と前走を結合
    df_all = pd.concat([df, df_split], axis=1)

    # target encording
    # df_all['平均クラス'] = np.nan
    # df_all['平均クラス'] = df_all['1クラス'].mean()

    # df_all['1クラス差'] = pd.to_numeric(df_all['平均クラス'].astype(str) + df_all['1クラス'].astype(str) + df_all['1過去着順'].astype(str), errors='coerce')

    # df_all = df_all.drop(['1クラス'], axis=1)

    # with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
    #     dist_dict = pickle.load(dd)
    # with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
    #     place_dict = pickle.load(dd)
    # with open(f'./pickle-dict/corner_dict{field_num}.pkl', "rb") as dd:
    #     coner_dict = pickle.load(dd)
    # with open(f'./pickle-dict/jhenka_dict{field_num}.pkl', "rb") as dd:
    #     jockey_dict = pickle.load(dd)
    # with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
    #     field_dict = pickle.load(dd)

    # df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    # df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    # df_all['1コーナー通過順'] = pd.to_numeric(df_all['1コーナー通過順'].astype(float).map(coner_dict), errors='coerce')
    # df_all['騎手変化'] = pd.to_numeric(df_all['騎手変化'].astype(float).map(jockey_dict), errors='coerce')
    # df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')

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
        df_split[sou+'場所変化'] = df_split[sou+'場所変化'].mask(df_split[sou+'場所'] == field_num, 1)
        df_split[sou+'フィールド変化'] = 0
        df_split[sou+'フィールド変化'] = df_split[sou+'フィールド変化'].mask(df_split[sou+'フィールド'] == df['フィールド'], 1)

        # 不要なカラムを削除
        df_split = df_split.drop([sou+'レース名'], axis=1)
        # df_split = df_split.drop([sou+'人気', sou+'レース名', sou+'場所', sou+'フィールド', sou+'馬場', sou+'距離', sou+'タイム', sou+'出走馬数', sou+'馬番', sou+'騎手', sou+'斤量', sou+'馬体重', sou+'体重増減', sou+'クラス'], axis=1)

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
        
    #     df_all[f'{i}クラス差'] = pd.to_numeric(df_all[f'{i}クラス差'].astype(float).map(class_dict), errors='coerce')
    #     df_all[f'{i}距離差'] = pd.to_numeric(df_all[f'{i}距離差'].astype(float).map(kyori_dict), errors='coerce')
    #     df_all[f'{i}場所変化'] = pd.to_numeric(df_all[f'{i}場所変化'].astype(float).map(basyo_dict), errors='coerce')
    #     df_all[f'{i}フィールド変化'] = pd.to_numeric(df_all[f'{i}フィールド変化'].astype(float).map(field_dict), errors='coerce')
    #     df_all[f'{i}人気差'] = pd.to_numeric(df_all[f'{i}人気差'].astype(float).map(ninki_dict), errors='coerce')


    # 上昇度カラム作成
    df_all['5過去着順'] = df_all['5過去着順'].astype(float)
    df_all['4過去着順'] = df_all['4過去着順'].astype(float)
    df_all['3過去着順'] = df_all['3過去着順'].astype(float)
    df_all['2過去着順'] = df_all['2過去着順'].astype(float)
    df_all['1過去着順'] = df_all['1過去着順'].astype(float)
    df_all['上昇度'] = (df_all['5過去着順'] - df_all['4過去着順']) + (df_all['4過去着順'] - df_all['3過去着順']) + (df_all['3過去着順'] - df_all['2過去着順']) + (df_all['2過去着順'] - df_all['1過去着順'])

    # Target Encording
    with open(f'./pickle-dict/kyori_dict{field_num}.pkl', "rb") as dd:
        dist_dict = pickle.load(dd)
    with open(f'./pickle-dict/basyo_dict{field_num}.pkl', "rb") as dd:
        place_dict = pickle.load(dd)
    # with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
    #     field_dict = pickle.load(dd)
    # with open(f'./pickle-dict/corner_dict{field_num}.pkl', "rb") as dd:
    #     coner_dict = pickle.load(dd)
    with open(f'./pickle-dict/jhenka_dict{field_num}.pkl', "rb") as dd:
        jockey_dict = pickle.load(dd)

    with open(f'./pickle-dict/field_dict{field_num}.pkl', "rb") as dd:
        field_dict = pickle.load(dd)
    # with open(f'./pickle-dict/1ninkisa_dict{field_num}.pkl', "rb") as dd:
    #     ninki_dict = pickle.load(dd)
    # with open(f'./pickle-dict/jokey_leading_dict{field_num}.pkl', "rb") as dd:
    #     leading_dict = pickle.load(dd)
    # with open(f'./pickle-dict/brad_dict{field_num}.pkl', "rb") as dd:
    #     brad_dict = pickle.load(dd)
    # with open(f'./pickle-dict/1class_dict{field_num}.pkl', "rb") as dd:
    #     classsa_dict = pickle.load(dd)
    with open(f'./pickle-dict/sire_dict{field_num}.pkl', "rb") as dd:
        sire_dict = pickle.load(dd)
    # with open(f'./pickle-dict/bms_dict{field_num}.pkl', "rb") as dd:
    #     bms_dict = pickle.load(dd)
    # with open(f'./pickle-dict/speed_dict{field_num}.pkl', "rb") as dd:
    #     speedrate_dict = pickle.load(dd)

    # # display(df_all[['1距離差', '1場所変化', '1フィールド', '1コーナー通過順', '騎手変化']])
    # df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    # df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    # df_all['1フィールド'] = pd.to_numeric(df_all['1フィールド'].astype(float).map(field_dict), errors='coerce')
    # df_all['1コーナー通過順'] = pd.to_numeric(df_all['1コーナー通過順'].astype(float).map(coner_dict), errors='coerce')
    # df_all['騎手変化'] = pd.to_numeric(df_all['騎手変化'].astype(float).map(jockey_dict), errors='coerce')
    # print(df_all[['1距離差', '1場所変化', '1フィールド変化', '1コーナー通過順', '騎手変化', '1スピード指数', '1人気差', '騎手',  '1クラス差', '父馬', '母父馬']].isnull().sum())

    df_all['1距離差'] = pd.to_numeric(df_all['1距離差'].astype(float).map(dist_dict), errors='coerce')
    df_all['1場所変化'] = pd.to_numeric(df_all['1場所変化'].astype(float).map(place_dict), errors='coerce')
    # df_all['1コーナー通過順'] = pd.to_numeric(df_all['1コーナー通過順'].astype(float).map(coner_dict), errors='coerce')
    df_all['騎手変化'] = pd.to_numeric(df_all['騎手変化'].astype(float).map(jockey_dict), errors='coerce')

    df_all['1フィールド変化'] = pd.to_numeric(df_all['1フィールド変化'].astype(float).map(field_dict), errors='coerce')
    # df_all['1人気差'] = pd.to_numeric(df_all['1人気差'].astype(float).map(ninki_dict), errors='coerce')
    # df_all['騎手'] = pd.to_numeric(df_all['騎手'].astype(float).map(leading_dict), errors='coerce')
    # df_all['血統'] = pd.to_numeric(df_all['血統'].astype(str).map(brad_dict), errors='coerce')
    # df_all['1クラス差'] = pd.to_numeric(df_all['1クラス差'].astype(float).map(classsa_dict), errors='coerce')
    df_all['父馬'] = pd.to_numeric(df_all['父馬'].astype(float).map(sire_dict), errors='coerce')
    # df_all['母父馬'] = pd.to_numeric(df_all['母父馬'].astype(float).map(bms_dict), errors='coerce')
    # df_all['1スピード指数'] = pd.to_numeric(df_all['1スピード指数'].astype(float).map(speedrate_dict), errors='coerce')

    result = pd.DataFrame()
    result['馬番'] = df_all['馬番']
    result['オッズ'] = df_all['オッズ']

    df_all['best着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].min(axis=1)
    df_all['bestスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].max(axis=1)

    df_all['av着差'] = df_all.loc[:, ['1着差', '2着差', '3着差', '4着差', '5着差']].mean(axis=1)
    df_all['avスピード指数'] = df_all.loc[:, ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数']].mean(axis=1)

    # 5用
    # df_all = df_all.drop(['オッズ', '2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '父馬', '母父馬', '1コーナー通過順', '間隔','1着差', '2着差', '3着差', '4着差', '5着差','1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数'], axis=1)

    # 6用
    df_all = df_all.drop(['オッズ', '2走', '3走', '4走', '5走', '母父馬', '騎手', '1コーナー通過順'], axis=1)
    # df_all から Nan を各列の中央値に置換する 4用
    # df_all = df_all.drop(['オッズ','2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順', '父馬', '母父馬', '1コーナー通過順'], axis=1)

    # df_all から Nan を各列の中央値に置換、削除する 3用
    # df_all = df_all.drop(['オッズ', '2走', '3走', '4走', '5走', '馬番', 'フィールド', '距離', '1過去着順', '2過去着順', '3過去着順', '4過去着順', '5過去着順',], axis=1)

    list_columns = ['場所', '過去着順', 'フィールド', '距離', 'タイム', '馬場', '出走馬数', '馬番', '人気', '騎手', '斤量', '後3F', '馬体重', '体重増減', '着差', '人気差', 'スピード指数']

    for i in range(1,5):
        k = i + 1
        for v in list_columns:
            df_all.fillna({f'{i}{v}': df_all[f'{k}{v}']}, inplace=True)
            df_all.fillna({f'{k}{v}': df_all[f'{i}{v}']}, inplace=True)

    # 説明変数をdataXに格納
    # display(df_all.columns.values)
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

    # results = results.sort_values(['Average'], ascending=[False])
    results = results.sort_values(['馬番'], ascending=[True])
    results['オッズ'] = result['オッズ']
    results['score'] = (results['Average'] - mean_df) / std_df
    results = results.sort_values('score', ascending=False)
    results['pro'] = results['score'].map('{:.1f}'.format)
    copy = results.copy()
    copy2 = results.copy()
    copy['pro'] = copy['pro'].astype(float).map(pro1_dict)
    copy2['pro'] = copy2['pro'].astype(str).map(pro3_dict)
    results['pro'] = results['pro'].astype(str).map(pro_dict)
    results.iat[0, results.columns.get_loc('pro')] = copy.iat[0, copy.columns.get_loc('pro')]
    results.iloc[1:3, results.columns.get_loc('pro')] = copy2.iloc[1:3, copy2.columns.get_loc('pro')]
    results['1_odds'] = 1 / results['pro'].astype(float)
    results['2_odds'] = 2 / results['pro'].astype(float)
    results['6_odds'] = 6 / results['pro'].astype(float)
    results['buy'] = 0
    results['buy'] = results['buy'].mask((results['pro'].astype(float) * results['オッズ'].astype(float)) >= 4, 1)
    results['buy'] = results['buy'].mask((results['pro'].astype(float) * results['オッズ'].astype(float)) >= 6, 2)
    results = results[['馬番', 'score', 'pro', '1_odds', '2_odds', '6_odds', 'buy']]



    # Z = result.iloc[:, 2:].values
    # for i in range(file_num):
    #     with open(f"{tuner_path}{i}-stack.pickle", mode="rb") as f:
    #             model = pickle.load(f)
    #     y_pred = model.predict(Z)

    #     result[f'stack{i}'] = y_pred

    # normal average
    # for i in range(file_num):
    #     mean_df = result[f'result{i}'].mean()
    #     std_df = result[f'result{i}'].std()
    #     result[f'nscore{i}'] = (result[f'result{i}'] - mean_df) / std_df
    result['Average'] = result.iloc[:, 2:].mean(axis=1)
    result['nscore'] = (result['Average'] - mean_df) / std_df

    # stack average
    # for i in range(file_num):
    #     mean_df = result[f'stack{i}'].mean()
    #     std_df = result[f'stack{i}'].std()
    #     result[f'score{i}'] = (result[f'stack{i}'] - mean_df) / std_df
    # result['s-Average'] = result.iloc[:, file_num+2:-2].mean(axis=1)
    # mean_df = result["s-Average"].mean()
    # std_df = result['s-Average'].std()
    # result['score'] = (result['s-Average'] - mean_df) / std_df

    # result = result.sort_values('s-Average', ascending=False)
    # results = result[['馬番', 'score']]
    result = result.sort_values('Average', ascending=False)
    result = result[['馬番', 'nscore']]
    result.rename(columns={'馬番': 'horse number'}, inplace=True)
    display(result)
    # display(results)
wait = input('press Enter')
