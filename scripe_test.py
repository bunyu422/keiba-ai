import pickle
import random
import re
import joblib
import numpy as np
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
import schedule
from sklearn.discriminant_analysis import StandardScaler
import torch
import Listwise
import Listwise_func
import betting
import function
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import logging
import datetime
import Learning

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)

# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)


# ブラウザ立ち上げ
# options = Options()
# # options = webdriver.FirefoxOptions()
# options.add_argument("--headless")#ヘッドレスの切替
# options.add_argument("--blink-settings=imagesEnabled=false")                                 # 画像を非表示にする。
# options.add_argument("--disable-background-networking")                                      # 拡張機能の更新、セーフブラウジングサービス、アップグレード検出、翻訳、UMAを含む様々なバックグラウンドネットワークサービスを無効にする。
# options.add_argument("--disable-blink-features=AutomationControlled")                        # navigator.webdriver=false となる設定。確認⇒　driver.execute_script("return navigator.webdriver")
# options.add_argument("--disable-default-apps")                                               # デフォルトアプリのインストールを無効にする。
# options.add_argument("--disable-dev-shm-usage")                                              # ディスクのメモリスペースを使う。DockerやGcloudのメモリ対策でよく使われる。
# options.add_argument("--disable-extensions")                                                 # 拡張機能をすべて無効にする。
# # options.add_argument("--disable-features=DownloadBubble")                                    # ダウンロードが完了したときの通知を吹き出しから下部表示(従来の挙動)にする。
# # options.add_argument('--disable-features=DownloadBubbleV2')                                  # `--incognito`を使うとき、ダイアログ(名前を付けて保存)を非表示にする。
# options.add_argument("--disable-features=Translate")                                         # Chromeの翻訳を無効にする。右クリック・アドレスバーから翻訳の項目が消える。
# options.add_argument("--disable-popup-blocking")                                             # ポップアップブロックを無効にする。
# # options.add_argument("--headless=new")                                                       # ヘッドレスモードで起動する。
# options.add_argument("--hide-scrollbars")                                                    # スクロールバーを隠す。
# options.add_argument("--ignore-certificate-errors")                                          # SSL認証(この接続ではプライバシーが保護されません)を無効
# # options.add_argument("--incognito")                                                          # シークレットモードで起動する。
# options.add_argument("--mute-audio")                                                         # すべてのオーディオをミュートする。
# options.add_argument("--no-default-browser-check")                                           # アドレスバー下に表示される「既定のブラウザとして設定」を無効にする。
# options.add_argument("--propagate-iph-for-testing")                                          # Chromeに表示される青いヒント(？)を非表示にする。
# options.add_argument("--start-maximized")                                                    # ウィンドウの初期サイズを最大化。--window-position, --window-sizeの2つとは併用不可
# # options.add_argument("--test-type=gpu")                                                      # アドレスバー下に表示される「Chrome for Testing~~」を非表示にする。
# # options.add_argument("--window-position=100,100")                                            # ウィンドウの初期位置を指定する。--start-maximizedとは併用不可
# # options.add_argument("--window-size=1600,1024")                                              # ウィンドウの初期サイズを設定する。--start-maximizedとは併用不可
# # options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])  # Chromeは自動テスト ソフトウェア~~ ｜ コンソールに表示されるエラー　を非表示
# # options.set_capability("browserVersion", "117")                                              # `--headless=new`を使うとき、コンソールに表示されるエラーを非表示にするための必須オプション

# # service = Service()
# # options.add_argument("--blink-settings=imagesEnabled=false")
# # options.add_argument("--window-size=1920,1080")  # ウィンドウサイズを指定
# # chrome_service = fs.Service(executable_path='/Users/XXXXXXXXX/Documents/Python/Driver/chromedriver')

# # options.add_argument("-headless")
# # driver = webdriver.Firefox(options=options)
# driver = webdriver.Chrome(options=options)
# driver.implicitly_wait(10)
# # wait = WebDriverWait(driver, 10)
# # url="https://www.ipat.jra.go.jp/sp/"
# url = "https://race.netkeiba.com/top/race_list.html?kaisai_date=20240922"
# driver.get(url)

def predict_multiple_races(model, df_all, feature_cols, cat_features,
                           context_num_features, context_cat_features,
                           group_col="レースID", device="cuda"):
    """
    複数レースをまとめて予測する。
    各レースごとにモデルへ入力し、予測結果を結合して返す。

    df_all : 全レースの特徴量を含む DataFrame
    group_col : レースを識別する列名（通常 'レースID'）
    """
    model.eval()
    all_results = []

    for race_id, df_race in df_all.groupby(group_col):
        df_race = df_race.copy()
        num_horses = len(df_race)

        # ---- 数値特徴 ----
        X = torch.tensor(df_race[feature_cols].values.astype(np.float32), dtype=torch.float32).to(device)

        # ---- カテゴリ特徴 ----
        if len(cat_features) > 0:
            cat_data = df_race[cat_features].values.astype(np.int64)
            for i, num_classes in enumerate(model.embedding_sizes):
                cat_data[:, i] = np.clip(cat_data[:, i], 0, num_classes - 1)
            cat_X = torch.tensor(cat_data, dtype=torch.long).to(device)
        else:
            cat_X = None

        # ---- コンテキスト数値特徴 ----
        if len(context_num_features) > 0:
            context_X = df_race[context_num_features].iloc[0].values.astype(np.float32)
            context_X = torch.tensor(np.tile(context_X, (num_horses, 1)), dtype=torch.float32).to(device)
        else:
            context_X = None

        # ---- コンテキストカテゴリ特徴 ----
        if len(context_cat_features) > 0:
            context_cat_data = df_race[context_cat_features].iloc[0].values.astype(np.int64).reshape(1, -1)
            for i, num_classes in enumerate(model.context_embedding_sizes):
                context_cat_data[:, i] = np.clip(context_cat_data[:, i], 0, num_classes - 1)
            context_cat_X = torch.tensor(np.tile(context_cat_data, (num_horses, 1)), dtype=torch.long).to(device)
        else:
            context_cat_X = None

        # ---- モデル予測 ----
        with torch.no_grad():
            pred_score = model(X, cat_X, context_X, context_cat_X)

        df_race["pred_score"] = pred_score.detach().cpu().numpy()
        all_results.append(df_race)

    # ---- 全レース結合 ----
    df_result = pd.concat(all_results, ignore_index=True)
    return df_result


def set_time(skip_list, url_race):
    time_list = []
    dict_race = []

    # ブラウザを起動する
    with webdriver.Chrome(options=options) as driver:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import re
        driver.get(url_race)

        # ページ全体が読み込まれるのを待つ（例: RaceList_DataList が出るまで最大10秒）
        blocks = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "RaceList_DataList"))
        )

        base_race_ids = []

        for block in blocks:
            li_items = block.find_elements(By.CSS_SELECTOR, "li.RaceList_DataItem.hasMovieLink")
            if not li_items:
                continue

            first_li = li_items[0]
            a_tag = first_li.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")

            match = re.search(r"race_id=(\d+)", href)
            if match:
                race_id = match.group(1)
                base_id = race_id[:-2]  # 末尾2桁を除く
                base_race_ids.append(base_id)

        print(base_race_ids)

        locations = []
        place_list = []
        # ブラウザでアクセスする
        driver.get(url_race)

        # 要素を取得
        el = driver.find_elements(By.CLASS_NAME, "RaceList_DataTitle")
        for i in el:

            # smallタグを取り除いたテキストだけ抜き出す
            text = i.get_attribute("innerText")

            # innerText は「3回\n 新潟 \n1日目」となるので strip/split で調整
            parts = text.split()
            # => ['3回', '新潟', '1日目']

            location = parts[1]  # "新潟"

            locations.append(location)

            print(location)



        # tableを取得(js反映)
        el=driver.find_elements(By.CLASS_NAME, "ItemTitle") #classでテーブルを指定

        for num, i in enumerate(el):
            text = i.text.strip()  # 要素のテキストを取得
            if "新馬" not in text:
                # 「新馬」が含まれていない要素だけ処理
                print("対象:", text)
                # ここに処理を書く
                dict_race.append(num)
                # tableを取得(js反映)

        el=driver.find_elements(By.CLASS_NAME, "RaceList_Itemtime") #classでテーブルを指定

        for i in range(len(el)):
            if i+1 in dict_race:
                time_list.append((dt.strptime(el[i].text, '%H:%M') - timedelta(minutes=3)).strftime("%H:%M"))

        for num, i in enumerate(locations, start=1):
            for j in range(sum(1 for x in dict_race if x <= 12*num and x > 12*(num-1))):
                place_list.append(i)
    
    print(dict_race)
    print(time_list)
    print(place_list)
    
# set_time([], url)
# print(betting.set_info())

# df = pd.read_csv('./csv/df_all_sonoda_2025.csv', index_col=0)
# print(df['父馬'].head(20))
# print(df.columns.values)
# df = pd.read_csv('./csv/kyoto_2012-2025.csv', index_col=0)
# print(df['レースID'].head(5))
# print(df['レースID'].tail(5))
# df = df[df['レースID'].astype(str).str[:4].astype(int) < 2017].reset_index(drop=True)
# df.to_csv('./csv/sonoda_2015-2025.csv', na_rep='NaN')

# Learning.scrape_payouts_combination('./csv/tokyo_payouts_2025.csv', '05', 2025, 2026)

# Learning.scraping_local('./csv/monbetu_2025.csv', '30', 2025, 2026)
# Learning.scraping_local('./csv/morioka_2015-2025.csv', '35', 2015, 2026)
# Learning.scraping_local('./csv/kasamatu_2015-2025.csv', '47', 2025, 2026)
# Learning.scraping_local('./csv/sonoda_2015-2025.csv', '50', 2025, 2026)
# Learning.scraping_local('./csv/nagoya_2025.csv', '48', 2025, 2026)
# Learning.scraping_local('./csv/mizusawa_2015-2024.csv', '36')
# Learning.scraping_local('./csv/hunabasi_2015-2025.csv', '43', 2025, 2026)
# Learning.scraping_local('./csv/saga_2015-2024.csv', '55')
# Learning.scraping_local('./csv/ooi_2015-2024.csv', '44')
# Learning.scraping_local('./csv/urawa_2015-2024.csv', '42')
# Learning.scraping_local('./csv/kanazawa_2015-2025.csv', '46', 2017, 2018)

# s = "マンハッタンカフェ ... 中3週 454kg"
# import pandas as pd
# print(pd.Series([s]).str.extract(r'(\d+)'))

# nakayama 無印 ~2016
# ver2 2017~2019
# ver3 2020~2022
# 4 2023~
# 5 24~
# csv_path = './csv/nakayama_2012_2025.csv' # 学習に使うcsvデータのパス
# file_num = 1
# df = pd.read_csv(csv_path, index_col=0)

# # print(df['レースID'].unique().tail(20))
# print(df['レースID'].unique()[-20:])

# --- 1. メインファイルを読み込み ---
# df = pd.read_csv('./csv/nakayama_2012_2025.csv')
# df = pd.read_csv('./csv/nakayama_2012_2025_all.csv')
# df = df[df['レースID'] == 202506040611].reset_index(drop=True)
# # print(df['2走'])
# with open("log.txt", "a", encoding="utf-8") as f:
#     f.write(df.sort_values(by=['馬番'])['2走'].to_string())

# --- 2. レースIDの先頭4文字が2017以上の行を削除 ---
# df = df[df['レースID'].astype(str).str[:4].astype(int) < 2017].reset_index(drop=True)

# # --- 3. 他のCSVを順に読み込み ---
# paths = [
#     './csv/nakayama_2012_2025_2.csv',
#     './csv/nakayama_2012_2025_3.csv',
#     './csv/nakayama_2012_2025_4.csv',
#     './csv/nakayama_2012_2025_5.csv'
# ]
# dfs = [pd.read_csv(p) for p in paths]

# # --- 4. 全部結合 ---
# df_all = pd.concat([df] + dfs, ignore_index=True)

# df = df_all[df_all['レースID'] == 202506040611].reset_index(drop=True)
# print(df)

# --- 5. 保存 ---
# df_all.to_csv('./csv/nakayama_2012_2025_all.csv', index=False)

# print("✅ 結合完了: ./csv/nakayama_2012_2025_all.csv に保存しました")
# print(f"総行数: {len(df_all)}")

# csv_path = f'./csv/df_all_nakayama_2025_2.csv'

# df = pd.read_csv(csv_path, index_col=0)
# df = df.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し
# # df = df[df['レースID'].astype(str).str[:4].astype(int) >= 2025].reset_index(drop=True)
# df = df[df['レースID'] == 202506040611]
# print(df.sort_values(by=['馬番'])[['馬番', '1斤量', '1馬場', '1タイム', '1フィールド', '1距離']])


# Learning.scraping('./csv/sapporo_2012-2024.csv', '01')
# Learning.scraping('./csv/hakodate_2012-2024.csv', '02')
Learning.scraping('./csv/hukushima_2012-2024.csv', '03', 2025, 2026)
# Learning.scraping('./csv/nigata_2012-2024.csv', '04')
# Learning.scraping('./csv/nakayama_2012_2025_5.csv', '06', 2024, 2026)
# Learning.scraping('./csv/tokyo_2025.csv', '05', 2025, 2026)
# Learning.scraping('./csv/chukyo_2012-2024.csv', '07')
# Learning.scraping('./csv/kyoto_2012-2025.csv', '08', 2025, 2026)
# Learning.scraping('./csv/hanshin_2012-2024.csv', '09')
# Learning.scraping('./csv/kokura_2012-2024.csv', '10')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# field = 'nakayama'
# field_num = 1
# csv_path = f"./csv/nakayama_2025.csv" # 学習に使うcsvデータのパス

# df = pd.read_csv(csv_path, index_col=0)
# df = df.reset_index(drop=True) # 行番号に重複があると.locがエラーを起こすので振り直し
# # print(pd.Series(sorted(df['レースID'].unique(), reverse=True)[:5]))
# df = df.replace(['', '未定', '除外', '取消', '失格', '中止'], 0)
# df['is_win'] = (df['着順'].astype(int) == 1).astype(int)
# # print(df['is_win'].head(30))
# df['場所'] = field_num
# # print(df.columns)
# # 今走の処理
# df = Learning.df_first_processing(df, field)
# # 過去走の処理
# df = Learning.df_big_past_processing(df, field, field_num)
# # 過去のレベル
# df = Learning.past_level(df)
# # 終了処理
# df = Learning.df_end_processing(df, 'a')
# # print(df.columns.values)
# # 逆数化
# df = Listwise.inversion(df)
# # カラム追加
# df = Listwise.append_col(df)
# df = Listwise.add_relative_features(df)
# fold = 0

# with open(f'./pickle-dict/sire_dict_nakayama3_fold{fold}.pkl', mode="rb") as f:
#     sire_mapping = pickle.load(f)


# # 列情報読み込み
# feature_cols = joblib.load("./pickle-dict/feature_cols2.pkl")
# embedding_cols = joblib.load("./pickle-dict/embedding_cols2.pkl")
# context_num_cols = joblib.load("./pickle-dict/context_num_cols2.pkl")
# context_cat_cols = joblib.load("./pickle-dict/context_cat_cols2.pkl")

# df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

# with open(f'./pickle-dict/jwin_dict_nakayama3_fold{fold}.pkl', "rb") as dd:
#     j_mapping = pickle.load(dd)

# # val/test は train 全体の mapping を使う
# df['騎手_te'] = df['騎手'].map(j_mapping).fillna(-1)

# scaler = joblib.load(f"./model/scaler_nakayama3_fold{fold}.pkl")
# df[Listwise.scale_cols] = scaler.transform(df[Listwise.scale_cols])

# # 欠損値補完
# df = Listwise.fill_nan(df, feature_cols)

# # カテゴリ列を数値化
# # df = Listwise.race_feature(df)
# category_mappings = joblib.load(f"./pickle-dict/category_mappings_nakayama3_fold{fold}.pkl")
# df = Listwise.race_feature_test(df, category_mappings)

# # print(df.head(30))

# embedding_sizes = []
# context_embedding_sizes = []
# # state_dict をロード
# state_dict = torch.load(f"./model/nakayama3_ranknet_{fold}.pth", map_location=device)

# # 通常のカテゴリ埋め込み
# i = 0
# while f"embeddings.{i}.weight" in state_dict:
#     num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
#     embedding_sizes.append(num_classes)
#     # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
#     i += 1

# # コンテキストカテゴリ埋め込み
# j = 0
# while f"context_embeddings.{j}.weight" in state_dict:
#     num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
#     context_embedding_sizes.append(num_classes)
#     # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
#     j += 1

# # 読み込み
# model = Listwise.ListNet(
#     embedding_sizes=embedding_sizes,
#     num_features=len(feature_cols),
#     context_embedding_sizes=context_embedding_sizes,
#     context_num_sizes=len(Listwise.context_num_cols),
#     emb_dim=32
# )
# model.load_state_dict(state_dict)
# model.to(device)
# model.eval()

# df = predict_multiple_races(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device=device)

# df.to_csv(f"./csv/nakayama3_result_fold{fold}.csv",na_rep='NaN')










# def scraping_local(csv_path, no, start, end):
#     df = pd.DataFrame()
#     # 複数の User-Agent を用意
#     USER_AGENTS = [
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
#         "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
#         "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
#         "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
#         "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36",
#     ]

#     # session = requests.Session()
#     # session.headers.update(headers)
#     for year in range(start, end):
#         for month in range(1, 13):
#             for day in range(1, 32):
#                 for race_no in range(1, 13):
#                     race_id = '201746031908'
#                     # race_id = "201530042201"
#                     url_race = 'https://nar.netkeiba.com/race/result.html?race_id={}&rf=race_list'.format(race_id)
#                     url_past = 'https://nar.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
#                     # ランダムにUser-Agentを選ぶ
#                     headers = {
#                         "User-Agent": random.choice(USER_AGENTS),
#                         "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
#                         "Accept-Encoding": "gzip, deflate, br",
#                         "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#                         "Referer": "https://www.google.com/",
#                         "Connection": "close",
#                     }
#                     try:
#                         response_race = requests.get(url_race, headers=headers)
#                         response_past = requests.get(url_past, headers=headers)
#                         df_result = pd.read_html(response_race.content)[0]
#                         df_past = pd.read_html(response_past.content)[0]
#                         soup = BeautifulSoup(response_race.content, 'html.parser')
#                         data1 = soup.find('div', class_='RaceData01').text
#                         data2 = soup.find('div', class_='RaceData02').text
#                         data3 = soup.find('tr', class_='Umatan').text
#                         data4 = soup.find('div', class_='RaceName').text
#                         a = data2[data2.find('新馬')+0: data2.find('新馬')+2]
#                         if a == '新馬':
#                             continue
#                         df_result_past = pd.merge(df_result, df_past, on='馬番')
#                         df_result_past['距離'] = re.findall(r'\d+', data1)[2]
#                         df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
#                         df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
#                         df_result_past['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')+0]
#                         df_result_past['馬単'] = data3
#                         df_result_past['レース名'] = data4.replace('\n', '')
#                         print(url_race)
#                         time.sleep(1)
#                     except Exception as e:
#                         if race_no == 1:
#                             print("no:"+url_race)
#                             break
#                         print(e)
#                         continue
#                     df_result_past['レースID'] = race_id
#                     df = pd.concat([df, df_result_past])


# scraping_local('./csv/kasamatu_2015-2025.csv', '47', 2017, 2018)