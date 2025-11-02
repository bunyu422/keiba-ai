from functools import reduce
import importlib
import pickle
from ssl import Options
import time
from bs4 import BeautifulSoup
import joblib
import pandas as pd
import requests
import torch
import Learning
import Listwise
import joblib
from bs4 import BeautifulSoup
import re
import pickle
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# 行を全表示（行の数）
pd.set_option("display.max_rows", None)
# 列を全表示（列の数）
pd.set_option("display.max_columns", None)
# セルの文字列を省略せずに全部表示
pd.set_option("display.max_colwidth", None)
# 小数点をすべて表示（指数表記なし）
pd.set_option('display.float_format', lambda x: f'{x:.16f}'.rstrip('0').rstrip('.'))

race_params = {
    # "hanshin": {"field_num": 4, "central": True},
    "tokyo": {"field_num": 2, "central": True},
    "nakayama": {"field_num": 1, "central": True},
    # "kyoto": {"field_num": 3, "central": True},
    "monbetu": {"field_num": 12, "central": False},
    "kasamatu": {"field_num": 20, "central": False},
    "sonoda": {"field_num": 22, "central": False},
    "nagoya": {"field_num": 21, "central": False},
    "saga": {"field_num": 25, "central": False},
    # "hunabasi": {"field_num": 16, "central": False},
}

def get_race_predict(race_id, field, odds):
    importlib.reload(Listwise)  # ←これでモジュール空間を完全初期化
    importlib.reload(Learning)  # ←これでモジュール空間を完全初期化

    params = race_params.get(field.lower())
    if params is None:
        raise ValueError(f"Unknown venue: {field}")
    
    dfs = get_race_info(race_id, field, params["field_num"], odds, params["central"])
    
    result_df = predict_lgb(dfs, field)
    
    result_df = result_df.sort_values('pred_score', ascending=False)
    print(result_df[['レースID', '馬番', 'pred_score']])

    return int(result_df.iloc[0]['馬番']), None

def add_pred_features(df, prefix="pred"):
    """
    複数の pred_score 列から統計特徴量を作成して DataFrame に追加する。

    Parameters
    ----------
    df : pd.DataFrame
        'pred_score_1', 'pred_score_2', ... のような列を含む DataFrame
    prefix : str
        新しく作る特徴量列のプレフィックス（例: 'pred' → 'pred_mean' など）

    Returns
    -------
    df : pd.DataFrame
        元の df に統計特徴量を追加した DataFrame
    """
    # pred_score系の列を抽出
    pred_cols = [c for c in df.columns if c.startswith("pred_score_")]
    if not pred_cols:
        raise ValueError("pred_score_ で始まる列が見つかりません。")

    # 各種特徴量を追加
    df[f"{prefix}_mean"]  = df[pred_cols].mean(axis=1)
    df[f"{prefix}_std"]   = df[pred_cols].std(axis=1)
    df[f"{prefix}_max"]   = df[pred_cols].max(axis=1)
    df[f"{prefix}_min"]   = df[pred_cols].min(axis=1)
    df[f"{prefix}_range"] = df[f"{prefix}_max"] - df[f"{prefix}_min"]
    df[f"{prefix}_var"]   = df[pred_cols].var(axis=1)
    df[f"{prefix}_median"] = df[pred_cols].median(axis=1)
    df[f"{prefix}_skew"]   = df[pred_cols].skew(axis=1)
    df[f"{prefix}_kurt"]   = df[pred_cols].kurt(axis=1)

    return df

def add_score_diff_features(df):
    """
    pred_score（モデルのスコア）を用いてレース内の相対的特徴量を追加する。
    """
    df = df.copy()
    new_rows = []

    for race_id, race in df.groupby('レースID'):
        race = race.sort_values(
            by=['pred_score', '馬番'],  # 第二ソートキーを指定
            ascending=[False, True]     # pred_scoreは降順、馬番は昇順
        ).reset_index(drop=True)
        
        mean_score = race['pred_score'].mean()
        std_score = race['pred_score'].std() if race['pred_score'].std() != 0 else 1e-6
        min_score = race['pred_score'].min()
        max_score = race['pred_score'].max()
        score_range = max_score - min_score if max_score != min_score else 1e-6
        
        # --- 差分・順位系 ---
        race['rank_in_race'] = range(1, len(race)+1)
        race['score_diff_prev'] = race['pred_score'].diff(-1)  # 下との差
        race['score_diff_next'] = race['pred_score'].diff()    # 上との差
        race['score_diff_top1'] = race['pred_score'].iloc[0] - race['pred_score']
        race['score_diff_top3_mean'] = race['pred_score'].iloc[:3].mean() - race['pred_score']
        
        # --- 統計・分布系 ---
        race['score_mean'] = mean_score
        race['score_std'] = std_score
        race['score_range'] = score_range
        race['score_cv'] = std_score / (mean_score + 1e-6)
        race['score_minus_mean'] = race['pred_score'] - mean_score
        race['score_minus_mean_std'] = (race['pred_score'] - mean_score) / std_score
        
        # --- 正規化・確率化 ---
        race['score_relative'] = (race['pred_score'] - min_score) / score_range
        exp_score = np.exp(race['pred_score'] - race['pred_score'].max())  # 安定化
        race['score_softmax'] = exp_score / exp_score.sum()
        race['score_z'] = (race['pred_score'] - mean_score) / std_score

        # --- 分布特性（レース単位） ---
        score_softmax = race['score_softmax'].values
        entropy = -np.sum(score_softmax * np.log(score_softmax + 1e-6))
        race['score_entropy'] = entropy
        race['score_top_mean'] = race['pred_score'].iloc[:3].mean()
        race['score_bottom_mean'] = race['pred_score'].iloc[-3:].mean()
        race['score_top_bottom_diff'] = race['score_top_mean'] - race['score_bottom_mean']
        race['score_top_ratio'] = race['pred_score'] / (race['pred_score'].iloc[0] + 1e-6)
        race['score_rank_gap_ratio'] = race['score_diff_prev'] / (race['pred_score'].abs() + 1e-6)

        new_rows.append(race)

    return pd.concat(new_rows, axis=0).reset_index(drop=True)


def reverse_category_mapping(df, category_mappings, cols=None):
    """
    カテゴリ値を元の文字列に戻す
    df: 変換済みのDataFrame
    category_mappings: 学習時に保存したカテゴリマッピング辞書
    cols: 対象の列（Noneなら category_mappings のキー全部）
    """
    if cols is None:
        cols = category_mappings.keys()
    
    for col in cols:
        if col not in df.columns:
            continue
        
        mapping = category_mappings[col]
        inv_map = {v: k for k, v in mapping.items()}  # 逆マップ作成
        
        # 未知カテゴリ値（n_train_categoriesなど）に対応する処理
        df[col] = df[col].map(inv_map).fillna("<UNK>").astype(str)
    
    return df


def get_race_info(race_id, field, field_num, odds, central):
    # ----------------------------
    # 基本設定
    # ----------------------------
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    df_shutuba = pd.DataFrame()

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
    response_race = session.get(url_race, headers=headers)
    response_past = session.get(url_past, headers=headers)
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
    # df_shutuba['オッズ'] = 0
    df_shutuba['人気'] = df1['人気']

    driver.quit()

    # ----------------------------
    # 取り消し馬削除
    # ----------------------------
    df = df_shutuba.copy()
    df = df[df['オッズ'] != '--']
    # df = df[df['人気'] != '--']

    # ----------------------------
    # Learning / Listwise 前処理
    # ----------------------------
    # ----------------------------
    # 場所列追加
    # ----------------------------
    df['場所'] = field_num
    df["騎手"] = df["騎手斤量"].str.extract(r'([^\s]+)\s*\d+\.?\d*$')
    df = Learning.df_first_processing(df, field)
    df = df.drop(['枠_x', '枠_y', '馬名_x', '馬名_y', '厩舎', '騎手斤量', '印_x', '印_y', '登録', '馬メモ切替', 'Unnamed: 9_level_1', 'グループ'], axis=1, errors="ignore")  # 必要な削除カラム
    df = Learning.df_big_past_processing(df, field, field_num)
    df = Learning.past_level(df)
    df = Learning.df_end_processing(df)
    df = Listwise.inversion(df)
    df = Listwise.append_col(df)
    df = Listwise.add_relative_features(df)

    if field == 'nakayama':
        field = 'nakayama3'
    
    # ----------------------------
    # dfをfold毎に分ける
    # ----------------------------
    
    # ----------------------------
    # ターゲットエンコーディング
    # 標準化・欠損値処理
    # ----------------------------

    dfs = [df.copy(deep=True) for _ in range(5)]
    feature_cols = joblib.load("./pickle-dict/lgb_cols.pkl")

    for fold, d in enumerate(dfs):
        d = mapping(d, '父馬', f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl')
        d = mapping(d, '騎手', f'./pickle-dict/jwin_dict_{field}_fold{fold}.pkl')
        d = Listwise.fill_nan(d, feature_cols)
        map_dict = joblib.load(f"./pickle-dict/category_mappings_{field}_fold{fold}.pkl")
        d = Listwise.race_feature_test(d, map_dict).round(10)

        # ✅ 修正: 更新したDataFrameをリストに戻す
        dfs[fold] = d

    return dfs

def mapping(df, target, path):
    df = df.copy()
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    df[f'{target}_te'] = df[target].map(mapping).fillna(-1)

    return df

def predict_listnet(
    df,
    field: str,
    fold: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        emb_dim=64
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return predict_new_data(model, df, feature_cols, embedding_cols, context_num_cols, context_cat_cols, device)

def predict_lgb(dfs, field):
    if field == 'nakayama':
        field = 'nakayama3'

    feature_cols = joblib.load("./pickle-dict/lgb_cols.pkl")
    type_list = ['reg-to-rank', 'reg-to-reg', 'rank-to-rank']
    # 各タイプごとに完全独立したDataFrameリストを作成
    dfs_list = [[df.copy(deep=True) for df in dfs] for _ in type_list]

    for model_type, dfs_l in zip(type_list, dfs_list): # for type in type_list:
        for fold, d in enumerate(dfs_l):
            model = joblib.load(f"./model/{field}_first_model_lgb_{model_type}_{fold}.pickle")
            d['pred_score'] = model.predict(d[feature_cols])

            # ✅ 修正: 更新したDataFrameをリストに戻す
            dfs_list[type_list.index(model_type)][fold] = d

    for i, dfs_l in enumerate(dfs_list):
        for k, d in enumerate(dfs_l):
            dfs_list[i][k] = add_score_diff_features(d).round(10) # 同率スコアがある場合順位が不定 修正必要

    feature_cols = joblib.load("./pickle-dict/lgb_cols_second.pkl")


    for model_type, dfs_l in zip(type_list, dfs_list): # for type in type_list:
        for fold, d in enumerate(dfs_l):
            for seed in range(1, 6):
                model = joblib.load(f"./model/{field}_second_model_lgb_{model_type}_seed{seed}_{fold}.pickle")
                dfs_list[type_list.index(model_type)][fold][f'result{seed}'] = model.predict(d[feature_cols])

            dfs_list[type_list.index(model_type)][fold]['pred_score'] = dfs_list[type_list.index(model_type)][fold][['result1', 'result2', 'result3', 'result4', 'result5']].mean(axis=1).round(10)


    feature_cols = joblib.load("./pickle-dict/stacking_feature_cols.pkl")
    result_df = dfs_list[0][0][['レースID', '馬番']].copy()
    for fold in range(5):
        df = dfs_list[0][fold].copy()
        df2 = dfs_list[1][fold].copy()
        df3 = dfs_list[2][fold].copy()

        # pred_score列をリネームして区別
        df  = df.rename(columns={'pred_score': 'pred_score_1'})
        df2 = df2.rename(columns={'pred_score': 'pred_score_3'})
        df3 = df3.rename(columns={'pred_score': 'pred_score_4'})

        df  = df.rename(columns={'result1': 'result1_1', 'result2': 'result2_1', 'result3': 'result3_1', 'result4': 'result4_1', 'result5': 'result5_1'})
        df2 = df2.rename(columns={'result1': 'result1_3', 'result2': 'result2_3', 'result3': 'result3_3', 'result4': 'result4_3', 'result5': 'result5_3'})
        df3 = df3.rename(columns={'result1': 'result1_5', 'result2': 'result2_5', 'result3': 'result3_5', 'result4': 'result4_5', 'result5': 'result5_5'})

        temp_dfs = [
            df[['レースID', '馬番', 'pred_score_1', 'result1_1', 'result2_1', 'result3_1', 'result4_1', 'result5_1']],
            df2[['レースID', '馬番', 'pred_score_3', 'result1_3', 'result2_3', 'result3_3', 'result4_3', 'result5_3']],
            df3[['レースID', '馬番', 'pred_score_4', 'result1_5', 'result2_5', 'result3_5', 'result4_5', 'result5_5']],
        ]
        # 結合（左から順に）
        temp_df = reduce(
            lambda left, right: pd.merge(left, right, on=['レースID', '馬番'], how='inner'),
            temp_dfs
        )

        temp_df = add_pred_features(temp_df).round(10)

        model = joblib.load(f"./model/stacking_model_lgb_fold{fold}.pickle")

        result_df[f'pred_score_{fold}'] = model.predict(temp_df[feature_cols])


    

    result_df = add_pred_features(result_df).round(10)
    feature_cols = joblib.load("./pickle-dict/stacking_fold_feature_cols.pkl")
    model = joblib.load("./model/stacking_fold_model_lgb.pickle")

    result_df['pred_score'] = model.predict(result_df[feature_cols])
    result_df = result_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])

    return result_df

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

    # df_new["pred_score"] = pred_score.detach().cpu().numpy()
    
    return pred_score.detach().cpu().numpy()

if __name__ == "__main__":
    # df = pd.read_csv(f'./csv/nakayama3_result_lgb_rank-to-rank_2025_0.csv', index_col=0)
    df = pd.read_csv(f'./csv/nakayama3_result_stacking_fold_2025.csv', index_col=0)
    # df = pd.read_csv('./csv/nakayama_kensyou_2025.csv', index_col=0)
    # df = pd.read_csv('./csv/df_all_nakayama_2025.csv', index_col=0)
    
    df = df.reset_index()
    # df = df[df['レースID'] == 202506040110]
    # print(df[['平均クラス', '平均ペース']])
    df = df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df['レースID'].unique().tolist())
    # df['馬番'] = df['馬番'].astype(int) + 1
    df = df[df['レースID'] == 202506040508]
    # print(df[['レースID', '馬番', '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数']])
    
    # feature_cols = joblib.load("./pickle-dict/lgb_cols.pkl")
    # print(feature_cols)
    # not_pop = joblib.load(f"./model/clf_ninki_input_nakayama3.pkl")
    # print(df.head(1)[not_pop])
    # print(df.head(16)[feature_cols])
    # with open("log.txt", "a", encoding="utf-8") as f:
    #     f.write(df.head(16).to_string())
    # レースごとに pred_score の順位をつける（降順）
    # df['rank'] = df.groupby('レースID')['pred_score'].rank(method='min', ascending=False)

    # # 1位と2位を抽出
    # top2 = df[df['rank'] <= 2]

    # top = df.loc[df.groupby('レースID')['pred_score'].idxmax()]
    # with open("log.txt", "a", encoding="utf-8") as f:
    #     f.write(top[['レースID', '馬番', 'pred_score']].to_string())
    # # # # print(df[['レースID', '馬番','pred_score']])
    # race_l = df['レースID'].unique().tolist()
    # print(df.head(16)[["pred_score_1", "pred_score_2", "pred_score_3", "pred_score_4", "pred_score_6"]])
    race_id, field, field_num, odds, central = 202506040508, 'nakayama', 1, 0, True
    odds = [i for i in range(len(df))]
    # race_l = [202506040304, 202506040304]
    # get_race_predict(race_id, field, field_num, odds, central, fold)
    # for race_id in race_l:
    #     race_id = int(race_id)
    #     # print(i)
    #     get_race_predict(race_id, field, field_num, odds, central)
    start = time.time()
    
    print(get_race_predict(race_id, field, odds))
    
    end = time.time()
    print("実行時間：", end - start)

    