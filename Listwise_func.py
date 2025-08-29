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
from sklearn.discriminant_analysis import StandardScaler
import torch
import Learning
import Listwise
import Listwise_test

warnings.simplefilter('ignore')

# 行・列ともに省略せず全て表示する設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def predict_new_data(model, df_new, feature_cols, cat_features, context_num_features, context_cat_features, device="cuda"):
    """
    新しいデータからスコアを予測する
    df_new: 新規レースの特徴量が入った DataFrame
    """
    model.eval()

    # ---- データをtensorに変換 (学習時と同じ処理を必ず行う) ----
    X = torch.tensor(df_new[feature_cols].values.astype(np.float32), dtype=torch.float32).to(device)
    cat_X = torch.tensor(df_new[cat_features].values.astype(np.int64), dtype=torch.long).to(device)

    num_horses = len(df_new)

    if len(context_num_features) > 0:
        context_X = df_new[context_num_features].iloc[0].values.astype(np.float32)
        context_X = torch.tensor(np.tile(context_X, (num_horses, 1)), dtype=torch.float32).to(device)
    else:
        context_X = None

    if len(context_cat_features) > 0:
        context_cat_X = df_new[context_cat_features].iloc[0].values.astype(np.int64)
        context_cat_X = torch.tensor(np.tile(context_cat_X, (num_horses, 1)), dtype=torch.long).to(device)
    else:
        context_cat_X = None

    # ---- モデルで予測 ----
    with torch.no_grad():
        pred_score = model(X, cat_X, context_X, context_cat_X)
    
    # CPUに戻して numpy化
    pred_score = pred_score.detach().cpu().numpy()

    # DataFrameに追加
    df_new["pred_score"] = pred_score

    return df_new


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

    # 今走の処理
    df = Learning.df_first_processing(df)
    # 過去走の処理
    df = Learning.df_big_past_processing(df)
    # 過去のレベル
    df = Learning.past_level(df)
    # 終了処理
    df = Learning.df_end_processing(df)
    # 逆数化
    df = Listwise.inversion(df)
    # カラム追加
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    with open('./pickle-dict/sire_dict2_fold2.pkl', mode="rb") as f:
        sire_mapping = pickle.load(f)

    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])

    Listwise.feature_cols = [col for col in df.columns if col not in ['レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

    df = Listwise.fill_nan(df, Listwise.feature_cols)

    embedding_cols = Listwise.embedding_init()

    feature_cols = [col for col in Listwise.feature_cols if col not in embedding_cols and col not in Listwise.common_cols]

    embedding_sizes = [df[col].nunique() + 1 for col in embedding_cols]  # 各カテゴリ列のクラス数
    context_embedding_sizes = [df[col].nunique() + 1 for col in Listwise.context_cat_cols]  # 各カテゴリ列のクラス数

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(torch.load("./model/tokyo_ranknet2_2.pth", map_location="cuda"))
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, Listwise.context_num_cols, Listwise.context_cat_cols)

    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=0.3)

    df = Listwise_test.filter_by_thresholds(df, ev_min=1.8, prob_min=0.0, odds_min=1.0, odds_max=15, ev_max=3)

    if len(df) == 0:
        return None
    else:
        print(df[["馬番", "オッズ", "softmax_score", "expected_value"]])
        return df.iloc[0]['馬番']