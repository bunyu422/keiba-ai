import time
import joblib
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
# GPU が使えるなら GPU に配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 競馬場別パラメータ設定
# ----------------------------
race_params = {
    "hanshin": {"field_num": 4, "ev_min": 2.0, "prob_min": 0.0, "odds_min": 4.0, "odds_max": np.inf, "ev_max": 4, "softmax_T": 0.6, "fold": 4, "central": True},
    "tokyo": {"field_num": 2, "ev_min": 2.0, "prob_min": 0.0, "odds_min": 3.0, "odds_max": np.inf, "ev_max": 4, "softmax_T": 0.2, "fold": 2, "central": True},
    "nakayama": {"field_num": 1, "ev_min": 1.5, "prob_min": 0.0, "odds_min": 4.0, "odds_max": np.inf, "ev_max": 5, "softmax_T": 0.2, "fold": 2, "central": True},
    "monbetu": {"field_num": 12, "ev_min": 2.0, "prob_min": 0.0, "odds_min": 1.0, "odds_max": np.inf, "ev_max": 3, "softmax_T": 0.6, "fold": 3, "central": False},
    "kasamatu": {"field_num": 20, "ev_min": 2.0, "prob_min": 0.0, "odds_min": 4.0, "odds_max": np.inf, "ev_max": np.inf, "softmax_T": 0.2, "fold": 0, "central": False},
    "sonoda": {"field_num": 22, "ev_min": 2.0, "prob_min": 0.0, "odds_min": 3.0, "odds_max": np.inf, "ev_max": 3.0, "softmax_T": 0.3, "fold": 1, "central": False},
    "nagoya": {"field_num": 21, "ev_min": 1.1, "prob_min": 0.0, "odds_min": 3.0, "odds_max": np.inf, "ev_max": 3.0, "softmax_T": 0.3, "fold": 3, "central": False},
}

race_params_wide = {
    "nakayama": {"field_num": 1, "ev_min": 1.8, "prob_min": 0.05, "odds_min": 4.0, "odds_max": np.inf, "ev_max": np.inf, "softmax_T": 0.6, "fold": 2}
}

def predict_new_data(model, df_new, feature_cols, cat_features, context_num_features, context_cat_features, device="cuda"):
    """
    新しいデータからスコアを予測する
    df_new: 新規レースの特徴量が入った DataFrame
    """
    model.eval()

    # ---- 数値特徴 ----
    X = torch.tensor(df_new[feature_cols].values.astype(np.float32), dtype=torch.float32).to(device)
    num_horses = len(df_new)

    # ---- カテゴリ特徴 (学習時embeddingに合わせてクリッピング) ----
    cat_data = df_new[cat_features].values.astype(np.int64)
    for i, num_classes in enumerate(model.embedding_sizes):  # 学習時のembedding_sizesをmodelに保持しておく
        cat_data[:, i] = np.clip(cat_data[:, i], 0, num_classes - 1)
    cat_X = torch.tensor(cat_data, dtype=torch.long).to(device)

    # ---- コンテキスト数値特徴 ----
    if len(context_num_features) > 0:
        context_X = df_new[context_num_features].iloc[0].values.astype(np.float32)
        context_X = torch.tensor(np.tile(context_X, (num_horses, 1)), dtype=torch.float32).to(device)
    else:
        context_X = None

    # ---- コンテキストカテゴリ特徴 ----
    if len(context_cat_features) > 0:
        context_cat_data = df_new[context_cat_features].iloc[0].values.astype(np.int64).reshape(1, -1)
        for i, num_classes in enumerate(model.context_embedding_sizes):
            context_cat_data[:, i] = np.clip(context_cat_data[:, i], 0, num_classes - 1)
        context_cat_X = torch.tensor(np.tile(context_cat_data, (num_horses, 1)), dtype=torch.long).to(device)
    else:
        context_cat_X = None

    # ---- モデルで予測 ----
    with torch.no_grad():
        pred_score = model(X, cat_X, context_X, context_cat_X)

    df_new["pred_score"] = pred_score.detach().cpu().numpy()
    
    return df_new

# ----------------------------
# 共通関数呼び出し用ラッパー
# ----------------------------
def select_horse(race_id: str, venue: str, odds: list):
    """
    競馬場ごとのパラメータを自動適用して共通関数を呼ぶ
    """
    params = race_params.get(venue.lower())
    if params is None:
        raise ValueError(f"Unknown venue: {venue}")
    
    params_wide = race_params_wide.get(venue.lower())
    
    df = get_race_info(race_id, field=venue.lower(), field_num=params["field_num"], odds=odds, central=params["central"])

    if params_wide is not None:
        
        return get_race_result(
            df,
            field=venue.lower(),
            field_num=params["field_num"],
            ev_min=params["ev_min"],
            prob_min=params["prob_min"],
            odds_min=params["odds_min"],
            odds_max=params["odds_max"],
            ev_max=params["ev_max"],
            softmax_T=params["softmax_T"],
            fold=params["fold"]
        ), \
        get_race_wide_result(
            df,
            field=venue.lower(),
            field_num=params_wide["field_num"],
            ev_min=params_wide["ev_min"],
            prob_min=params_wide["prob_min"],
            odds_min=params_wide["odds_min"],
            odds_max=params_wide["odds_max"],
            ev_max=params_wide["ev_max"],
            softmax_T=params_wide["softmax_T"],
            fold=params_wide["fold"]
        )
    else:
        return get_race_result(
            df,
            field=venue.lower(),
            field_num=params["field_num"],
            ev_min=params["ev_min"],
            prob_min=params["prob_min"],
            odds_min=params["odds_min"],
            odds_max=params["odds_max"],
            ev_max=params["ev_max"],
            softmax_T=params["softmax_T"],
            fold=params["fold"]
        ), None

def get_race_result(
    df,
    field: str,
    field_num: int,
    ev_min: float,
    prob_min: float,
    odds_min: float,
    odds_max: float,
    ev_max: float,
    softmax_T: float,
    fold: int
):
    """
    共通化した競馬レース結果取得関数。
    
    Parameters
    ----------
    race_id : str
        netkeibaのレースID
    field : str
        競馬場名 (hanshin, tokyo, nakayamaなど)
    odds : float
        オッズ
    field_num : int
        場所番号 (1, 2, 4など)
    ev_min : float
        期待値最小閾値
    odds_min : float
        オッズ最小閾値
    odds_max : float
        オッズ最大閾値
    ev_max : float
        期待値最大閾値
    softmax_T : float
        softmax温度
    fold : int
        交差検証用fold
        
    Returns
    -------
    int or None
        選択された馬番、または条件に合致しない場合 None
    """


    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    # ----------------------------
    # 父馬マッピング
    # ----------------------------
    pkl_path = f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl'
    with open(pkl_path, "rb") as f:
        sire_mapping = pickle.load(f)
    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    # ----------------------------
    # 標準化・欠損値処理
    # ----------------------------
    # with open('log.txt', 'a', encoding='utf-8') as f:
    #     print(df, file=f)
    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])
    df = Listwise.fill_nan(df, feature_cols)

    # ----------------------------
    # 場所列追加
    # ----------------------------
    df['場所'] = field_num

    # ----------------------------
    # カテゴリ列数値化
    # ----------------------------
    df = Listwise.race_feature(df)

    # ----------------------------
    # モデルロード・推論
    # ----------------------------
    
    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    model_path = f"./model/{field}_ranknet_{fold}.pth"
    state_dict = torch.load(model_path, map_location=device)

    embedding_sizes = []
    context_embedding_sizes = []

    # 通常のカテゴリ埋め込み
    i = 0
    while f"embeddings.{i}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
        embedding_sizes.append(num_classes)
        # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
        i += 1

    # コンテキストカテゴリ埋め込み
    j = 0
    while f"context_embeddings.{j}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
        context_embedding_sizes.append(num_classes)
        # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
        j += 1

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

    # ----------------------------
    # 馬番修正
    # ----------------------------
    df['馬番'] = df['馬番'].astype(int) + 1

    # ----------------------------
    # 期待値最大選択 & 閾値フィルタ
    # ----------------------------
    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=softmax_T)
    print(df[["馬番","オッズ","softmax_score","expected_value"]], flush=True)
    df = Listwise_test.filter_by_thresholds(df, ev_min=ev_min, prob_min=prob_min, odds_min=odds_min, odds_max=odds_max, ev_max=ev_max)

    if len(df) == 0:
        return None
    else:
        return int(df.iloc[0]['馬番'])
    
def get_race_wide_result(
    df,
    field: str,
    field_num: int,
    ev_min: float,
    prob_min: float,
    odds_min: float,
    odds_max: float,
    ev_max: float,
    softmax_T: float,
    fold: int
):
    """
    共通化した競馬レース結果取得関数。
    
    Parameters
    ----------
    race_id : str
        netkeibaのレースID
    field : str
        競馬場名 (hanshin, tokyo, nakayamaなど)
    odds : float
        オッズ
    field_num : int
        場所番号 (1, 2, 4など)
    ev_min : float
        期待値最小閾値
    odds_min : float
        オッズ最小閾値
    odds_max : float
        オッズ最大閾値
    ev_max : float
        期待値最大閾値
    softmax_T : float
        softmax温度
    fold : int
        交差検証用fold
        
    Returns
    -------
    int or None
        選択された馬番、または条件に合致しない場合 None
    """


    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    # ----------------------------
    # 父馬マッピング
    # ----------------------------
    pkl_path = f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl'
    with open(pkl_path, "rb") as f:
        sire_mapping = pickle.load(f)
    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    # ----------------------------
    # 標準化・欠損値処理
    # ----------------------------
    # with open('log.txt', 'a', encoding='utf-8') as f:
    #     print(df, file=f)
    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])
    df = Listwise.fill_nan(df, feature_cols)

    # ----------------------------
    # 場所列追加
    # ----------------------------
    df['場所'] = field_num

    # ----------------------------
    # カテゴリ列数値化
    # ----------------------------
    df = Listwise.race_feature(df)

    # ----------------------------
    # モデルロード・推論
    # ----------------------------
    
    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    model_path = f"./model/{field}_ranknet_{fold}.pth"
    state_dict = torch.load(model_path, map_location=device)

    embedding_sizes = []
    context_embedding_sizes = []

    # 通常のカテゴリ埋め込み
    i = 0
    while f"embeddings.{i}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
        embedding_sizes.append(num_classes)
        # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
        i += 1

    # コンテキストカテゴリ埋め込み
    j = 0
    while f"context_embeddings.{j}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
        context_embedding_sizes.append(num_classes)
        # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
        j += 1

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

    # ----------------------------
    # 馬番修正
    # ----------------------------
    df['馬番'] = df['馬番'].astype(int) + 1

    # ----------------------------
    # 期待値最大選択 & 閾値フィルタ
    # ----------------------------
    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=softmax_T, bet_type="ワイド")
    print(df[["馬番","オッズ","softmax_score","expected_value"]], flush=True)
    # 先頭評価
    # main_candidates = df.head(1)
    # main_candidates = Listwise_test.filter_by_thresholds(main_candidates, ev_min=ev_min, prob_min=prob_min, odds_min=odds_min, odds_max=odds_max, ev_max=ev_max)

    # if len(main_candidates) == 0:
    #     return None
    # else:
    #     return df["馬番"].astype(int).tolist()
    return df["馬番"].astype(int).tolist()
    
def get_race_info(race_id, field, field_num, odds, central):
    # ----------------------------
    # 基本設定
    # ----------------------------
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    df_shutuba = pd.DataFrame()

    headers = {"User-Agent": "Mozilla/5.0"}

    if central:
        url_race = 'https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
        url_past = 'https://race.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    else:
        url_race = 'https://nar.netkeiba.com/race/shutuba.html?race_id={}&rf=shutuba_submenu'.format(race_id)
        url_past = 'https://nar.netkeiba.com/race/shutuba_past.html?race_id={}&rf=shutuba_submenu'.format(race_id)
    print(url_race)

    # ----------------------------
    # 過去・現在データ取得
    # ----------------------------
    response_race = requests.get(url_race, headers=headers)
    response_past = requests.get(url_past, headers=headers)
    df_now = pd.read_html(response_race.content)[0]
    df_past = pd.read_html(response_past.content)[0]
    df_now.columns = df_now.columns.droplevel()
    df_result_past = pd.merge(df_now, df_past, on='馬番')

    # ----------------------------
    # BeautifulSoupでレース情報抽出
    # ----------------------------
    soup = BeautifulSoup(response_race.content, "html.parser")
    data1 = soup.find('div', class_='RaceData01').text
    data2 = soup.find('div', class_='RaceData02').text
    df_result_past['距離'] = int(re.findall(r'\d+', data1)[2])
    df_result_past['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_result_past['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_result_past['出走頭数'] = int(data2[data2.find('頭')-2: data2.find('頭')])
    df_result_past['レースID'] = race_id

    # ----------------------------
    # Seleniumで人気取得
    # ----------------------------
    options = Options()
    options.add_argument("--headless")
    options.add_argument('--log-level=3')
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    driver.get(url_race)
    el = driver.find_element(By.CLASS_NAME, "RaceTableArea")
    html = el.get_attribute("outerHTML")
    df1 = pd.read_html(html)[0]
    df1.reset_index(inplace=True, drop=True)
    df1.columns = df1.columns.droplevel()

    df_shutuba = pd.concat([df_shutuba, df_result_past])
    df_shutuba['オッズ'] = odds
    df_shutuba['人気'] = df1['人気']

    driver.quit()

    # ----------------------------
    # 取り消し馬削除
    # ----------------------------
    df = df_shutuba.copy()
    df = df[df['オッズ'] != '--']

    # ----------------------------
    # Learning / Listwise 前処理
    # ----------------------------
    df = Learning.df_first_processing(df, field)
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', '登録', '馬メモ切替', 'Unnamed: 9_level_1', 'グループ'], axis=1, errors="ignore")  # 必要な削除カラム
    df = Learning.df_big_past_processing(df, field, field_num)
    df = Learning.past_level(df)
    df = Learning.df_end_processing(df)
    df = Listwise.inversion(df)
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(df.to_string())

    return df


def hanshin(race_id, odds):
    field = 'hanshin'
    field_num = 4

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

    # display(df.columns)

    # 取り消し馬を削除
    df = df_shutuba.copy()
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    # 今走の処理
    df = Learning.df_first_processing(df, field)
    # いらないカラムを消す
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1, errors="ignore")
    # 過去走の処理
    df = Learning.df_big_past_processing(df, field, field_num)
    # 過去のレベル
    df = Learning.past_level(df)
    # 終了処理
    df = Learning.df_end_processing(df)
    # 逆数化
    df = Listwise.inversion(df)
    # カラム追加
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    with open('./pickle-dict/sire_dict_hanshin_fold4.pkl', mode="rb") as f:
        sire_mapping = pickle.load(f)

    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])

    # 欠損値補完
    df = Listwise.fill_nan(df, feature_cols)

    # カラム追加
    df['場所'] = 4

    # カテゴリ列を数値化
    df = Listwise.race_feature(df)

    embedding_sizes = []
    context_embedding_sizes = []
    # state_dict をロード
    state_dict = torch.load("./model/hanshin_ranknet_4.pth", map_location=device)

    # 通常のカテゴリ埋め込み
    i = 0
    while f"embeddings.{i}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
        embedding_sizes.append(num_classes)
        # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
        i += 1

    # コンテキストカテゴリ埋め込み
    j = 0
    while f"context_embeddings.{j}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
        context_embedding_sizes.append(num_classes)
        # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
        j += 1

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

    df['馬番'] = df['馬番'].astype(int) + 1

    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=0.6)
    print(df[["馬番","オッズ","softmax_score","expected_value"]], flush=True)

    df = Listwise_test.filter_by_thresholds(df, ev_min=2.0, prob_min=0.0, odds_min=4.0, odds_max=np.inf, ev_max=4)

    if len(df) == 0:
        return None
    else:
        return df.iloc[0]['馬番']

def tokyo(race_id, odds):
    field = 'tokyo'
    field_num = 2

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

    # display(df.columns)

    # 取り消し馬を削除
    df = df_shutuba.copy()
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    # 今走の処理
    df = Learning.df_first_processing(df, field)
    # いらないカラムを消す
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1, errors="ignore")
    # 過去走の処理
    df = Learning.df_big_past_processing(df, field, field_num)
    # 過去のレベル
    df = Learning.past_level(df)
    # 終了処理
    df = Learning.df_end_processing(df)
    # 逆数化
    df = Listwise.inversion(df)
    # カラム追加
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    with open('./pickle-dict/sire_dict_tokyo_fold2.pkl', mode="rb") as f:
        sire_mapping = pickle.load(f)

    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])

    # 欠損値補完
    df = Listwise.fill_nan(df, feature_cols)

    # カラム追加
    df['場所'] = 2

    # カテゴリ列を数値化
    df = Listwise.race_feature(df)

    # embedding_sizes = [df[col].nunique() + 5 for col in embedding_cols]  # 各カテゴリ列のクラス数
    # context_embedding_sizes = [df[col].nunique() + 5 for col in Listwise.context_cat_cols]  # 各カテゴリ列のクラス数

    embedding_sizes = []
    context_embedding_sizes = []
    # state_dict をロード
    state_dict = torch.load("./model/tokyo_ranknet_2.pth", map_location=device)

    # 通常のカテゴリ埋め込み
    i = 0
    while f"embeddings.{i}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
        embedding_sizes.append(num_classes)
        # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
        i += 1

    # コンテキストカテゴリ埋め込み
    j = 0
    while f"context_embeddings.{j}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
        context_embedding_sizes.append(num_classes)
        # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
        j += 1

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

    df['馬番'] = df['馬番'].astype(int) + 1

    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=0.2)
    print(df[["馬番","オッズ","softmax_score","expected_value"]], flush=True)

    df = Listwise_test.filter_by_thresholds(df, ev_min=2.0, prob_min=0.0, odds_min=3.0, odds_max=np.inf, ev_max=4)

    if len(df) == 0:
        return None
    else:
        return df.iloc[0]['馬番']
    
def nakayama(race_id, odds):
    field = 'nakayama'
    field_num = 1

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

    # 取り消し馬を削除
    df = df_shutuba.copy()
    indexNames = df[df['オッズ'] != '--']
    df = indexNames

    # 今走の処理
    df = Learning.df_first_processing(df, field)
    # いらないカラムを消す
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', '登録', '馬メモ切替', 'Unnamed: 9_level_1'], axis=1, errors="ignore")
    # 過去走の処理
    df = Learning.df_big_past_processing(df, field, field_num)
    # 過去のレベル
    df = Learning.past_level(df)
    # 終了処理
    df = Learning.df_end_processing(df)
    # 逆数化
    df = Listwise.inversion(df)
    # カラム追加
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    with open('./pickle-dict/sire_dict_nakayama_fold2.pkl', mode="rb") as f:
        sire_mapping = pickle.load(f)

    # 列情報読み込み
    feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
    embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
    context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
    context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

    df['父馬_te'] = df['父馬'].map(sire_mapping).fillna(-1)

    scaler = StandardScaler()
    df[Listwise.scale_cols] = scaler.fit_transform(df[Listwise.scale_cols])

    # 欠損値補完
    df = Listwise.fill_nan(df, feature_cols)

    # カラム追加
    df['場所'] = 1

    # カテゴリ列を数値化
    df = Listwise.race_feature(df)

    embedding_sizes = []
    context_embedding_sizes = []
    # state_dict をロード
    state_dict = torch.load("./model/nakayama_ranknet_2.pth", map_location=device)

    # 通常のカテゴリ埋め込み
    i = 0
    while f"embeddings.{i}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"embeddings.{i}.weight"].shape
        embedding_sizes.append(num_classes)
        # print(f"embeddings.{i}: {num_classes} classes, {emb_dim} dim")
        i += 1

    # コンテキストカテゴリ埋め込み
    j = 0
    while f"context_embeddings.{j}.weight" in state_dict:
        num_classes, emb_dim = state_dict[f"context_embeddings.{j}.weight"].shape
        context_embedding_sizes.append(num_classes)
        # print(f"context_embeddings.{j}: {num_classes} classes, {emb_dim} dim")
        j += 1

    # 読み込み
    model = Listwise.ListNet(
        embedding_sizes=embedding_sizes,
        num_features=len(feature_cols),
        context_embedding_sizes=context_embedding_sizes,
        context_num_sizes=len(Listwise.context_num_cols),
        emb_dim=16
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    df = predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

    df['馬番'] = df['馬番'].astype(int) + 1

    df = Listwise_test.pick_ev_max_per_race(df, race_col="レースID", T=0.2)
    print(df[["馬番","オッズ","softmax_score","expected_value"]], flush=True)

    df = Listwise_test.filter_by_thresholds(df, ev_min=1.5, prob_min=0.0, odds_min=4.0, odds_max=np.inf, ev_max=5)

    if len(df) == 0:
        return None
    else:
        return df.iloc[0]['馬番']

def all_place(race_id, odds):

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
    

if __name__ == "__main__":
    start = time.time()
    
    result = select_horse(202506040202, 'nakayama', [43.6, 2.9, 18.7, 172.1, 52.1, 10.0, 146.0, 43.6, 2.9, 18.7, 172.1, 52.1, 10.0, 146.0, 2, 3])

    end = time.time()
    print("実行時間：", end - start)