from itertools import combinations
import itertools
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

save_mapping_path = f'./csv/tokyo_result_lgb_test_0.csv'
csv_path = f'./csv/tokyo_result_lgb_2025_0.csv'
df = pd.read_csv(save_mapping_path, index_col=0).reset_index(drop=True)
df2 = pd.read_csv(csv_path, index_col=0).reset_index(drop=True)

# print(df[['pred_score', 'オッズ']].head(30))

from sklearn.isotonic import IsotonicRegression

def fit_isotonic(preds, is_win):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(preds, is_win)
    return iso

def apply_isotonic(iso, preds):
    return iso.predict(preds)

def fit_bin_calibration(preds, is_win, n_bins=20):
    df = pd.DataFrame({'pred': preds, 'win': is_win})
    df = df.sort_values('pred')

    df['bin'] = pd.qcut(df['pred'], q=n_bins, duplicates='drop')
    bin_stats = df.groupby('bin')['win'].mean()

    # bin の境界（左側は閉区間）
    bin_edges = [b.left for b in bin_stats.index] + [bin_stats.index[-1].right]

    return bin_edges, bin_stats.values

def apply_bin_calibration(preds, bin_edges, bin_probs):
    calibrated = np.zeros_like(preds)

    for i, p in enumerate(preds):
        # 各 bin に入れて勝率割当
        idx = np.searchsorted(bin_edges, p, side='right') - 1
        idx = np.clip(idx, 0, len(bin_probs) - 1)
        calibrated[i] = bin_probs[idx]

    return calibrated

def scores_to_support_rate(scores, temperature=0.5):
    """
    同じレース内のモデルスコアを softmax して支持率に変換
    """
    scores = np.array(scores)
    exp_scores = np.exp(scores / temperature)
    probs = exp_scores / exp_scores.sum()
    return probs

def pseudo_odds(scores, R=0.8):
    """
    scores: 同じレースに出走する馬のモデルスコア（リスト or np.array）
    R: JRA の配当率
    """
    support_rates = scores_to_support_rate(scores)
    odds = 1.0 / support_rates * R  # 配当率を掛けて疑似オッズ
    return odds

# --- レースごとに疑似オッズを計算 ---
def add_racewise_pseudo_odds(df, score_col='pred_score', race_col='レースID', R=0.8, temperature=1.0):
    """
    df: DataFrame
    score_col: モデルスコア列
    race_col: レースID列
    R: 配当率
    temperature: softmax 温度パラメータ
    """
    def compute_odds(group):
        scores = group[score_col].values
        odds = pseudo_odds(scores, R=R)
        return pd.Series(odds, index=group.index)

    df['疑似オッズ'] = df.groupby(race_col, group_keys=False).apply(compute_odds)
    return df

import pandas as pd
from scipy.stats.mstats import winsorize

from scipy.stats.mstats import winsorize

def calculate_roi_from_odds(df, bet_amount=100, winsor_limits=(0.0, 0.0)):
    df2 = df.copy()
    
    # 払戻列を作る（的中時はオッズ×bet_amount、外れは0）
    df2['payout'] = df2['単勝オッズ']
    df2['bet'] = bet_amount

    # 全体で winsorize
    df2['payout_w'] = winsorize(df2['payout'], limits=winsor_limits)
    df2['bet_w'] = winsorize(df2['bet'], limits=winsor_limits)

    # odds_bin作成
    bins = [0,3,7,20,999]
    labels = ["〜3倍", "3〜7倍", "7〜20倍", "20倍〜"]
    df2['odds_bin'] = pd.cut(df2['オッズ'], bins=bins, labels=labels, right=False)

    # binごとに集計
    summary = df2.groupby('odds_bin').agg(
        count=('オッズ', 'size'),
        hit_rate=('is_win', 'mean'),
        hit_count=('is_win', 'sum'),
        total_payout=('payout_w', 'sum'),
        total_bet=('bet_w', 'sum')
    )
    summary['ROI'] = summary['total_payout'] / summary['total_bet']
    return summary.reset_index()

def calculate_roi_from_odds_fuku(df, bet_amount=100, winsor_limits=(0.00, 0.00)):
    df2 = df.copy()
    
    # 払戻列を作る（的中時はオッズ×bet_amount、外れは0）
    df2['payout'] = df2['複勝払戻']
    df2['bet'] = bet_amount

    # 全体で winsorize
    df2['payout_w'] = winsorize(df2['payout'], limits=winsor_limits)
    df2['bet_w'] = winsorize(df2['bet'], limits=winsor_limits)

    # odds_bin作成
    bins = [0,3,7,20,999]
    labels = ["〜3倍", "3〜7倍", "7〜20倍", "20倍〜"]
    df2['odds_bin'] = pd.cut(df2['オッズ'], bins=bins, labels=labels, right=False)

    # binごとに集計
    summary = df2.groupby('odds_bin').agg(
        count=('オッズ', 'size'),
        hit_rate=('複勝_hit', 'mean'),
        hit_count=('複勝_hit', 'sum'),
        total_payout=('payout_w', 'sum'),
        total_bet=('bet_w', 'sum')
    )
    summary['ROI'] = summary['total_payout'] / summary['total_bet']
    return summary.reset_index()

def select_top_with_odds(df, score_col="pred_score", odds_col="オッズ"):
    selected_idx = []

    # レースごとに処理
    for race_id, g in df.groupby("レースID"):
        # スコア順にソート（降順）
        g_sorted = g.sort_values(score_col, ascending=False)

        # 7倍以上の馬を上から探す
        hit = g_sorted[g_sorted[odds_col] >= 7]

        if len(hit) > 0:
            # 最初に見つかった馬
            selected_idx.append(hit.index[0])
        else:
            # 全部7倍未満 → 次点を選ぶロジックに応じて以下
            # 「スコアトップが7倍未満でも次点を選ぶ」なら
            # 2番手を選ぶ(存在すれば)
            selected_idx.append(g_sorted.index[0])

    return df.loc[selected_idx]

def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)     

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader の worker 初期化用
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

def select_top3_over7(df_pred, race_col="レースID", score_col="pred_score"):
    results = []

    for race_id, g in df_pred.groupby(race_col):

        # スコアで降順ソート
        g_sorted = g.sort_values(score_col, ascending=False)

        # オッズ7倍以上を抽出
        g_filtered = g_sorted[g_sorted["オッズ"] >= 7]

        # 3頭未満なら補填する（←ここが重要）
        if len(g_filtered) < 3:
            need = 3 - len(g_filtered)

            # まだ選んでいない馬で下の順位から補充
            g_remain = g_sorted[~g_sorted.index.isin(g_filtered.index)]
            g_add = g_remain.head(need)

            g_final = pd.concat([g_filtered, g_add])
        else:
            g_final = g_filtered.head(3)

        g_final[race_col] = race_id
        results.append(g_final)

    return pd.concat(results)

def bootstrap_wide_roi(
    df_pred,              # スコア付きデータ
    df_payout,            # 払戻データ（券種, 馬番(1-2形式), 払い戻し金額）
    race_col="レースID",
    score_col="pred_score",
    n_boot=10000,
    stake=100
):

    # --- 1. レース内でスコア順に並べ、上位3頭だけ抽出 ---
    sel = (
        df_pred.sort_values([race_col, score_col], ascending=[True, False])
        .groupby(race_col)
        .head(3)
    )

    # --- 2. race_id → [馬番リスト] を作成 ---
    sel["馬番"] = sel["馬番"].astype(int)
    horse_lists = sel.groupby(race_col)["馬番"].apply(list)

    # --- 2. 購入対象データ（race_id, 馬番） ---
    # df_bets = sel[[race_col, "馬番"]].copy()
    # df_bets["馬番"] = df_bets["馬番"].astype(str)  # 払戻の馬番と型を合わせる

    # --- 3. レースごとに "1位 → 2,3位" のワイド2点だけ作る ---
    bets = []
    for race_id, horses in horse_lists.items():
        if len(horses) >= 2:
            main = horses[0]
            others = horses[1:2]
            for o in others:
                pair = "-".join(map(str, sorted([main, o])))
                bets.append((race_id, pair))

    # --- 各レースで「上位3頭BOX（3点）」を生成 ---
    # bets = []
    # for race_id, horses in horse_lists.items():
    #     if len(horses) >= 3:
    #         top3 = horses[:3]  # 上位3頭
    #         # 全ての2頭組み合わせ（3C2 = 3点）
    #         for h1, h2 in combinations(top3, 2):
    #             pair = "-".join(map(str, sorted([h1, h2])))
    #             bets.append((race_id, pair))

    df_bets = pd.DataFrame(bets, columns=[race_col, "馬番"])

    if df_bets.empty:
        return {
            "roi_mean": 0,
            "roi_ci": (0, 0),
            "acc_mean": 0,
            "roi_list": [],
            "acc_list": []
        }

    # --- 4. ワイド払戻だけ抽出（馬番は "1-2" のような文字列） ---
    payout_wide = df_payout.query("券種 == 'ワイド'")[
        [race_col, "馬番", "払い戻し金額"]
    ]
    
    payout_wide["馬番"] = payout_wide["馬番"].astype(str)

    # --- 5. ベット情報と払戻を結合 ---
    df_merged = df_bets.merge(payout_wide, on=[race_col, "馬番"], how="left")
    # print(df_merged)

    df_merged["hit"] = df_merged["払い戻し金額"].notna().astype(int)
    df_merged["payout"] = df_merged["払い戻し金額"].fillna(0)

    # --- 6. ブートストラップ ---
    roi_list = []
    acc_list = []

    races = df_merged[race_col].unique()

    for _ in range(n_boot):
        sampled_races = np.random.choice(races, size=len(races), replace=True)
        sampled = df_merged[df_merged[race_col].isin(sampled_races)]

        total_bet = len(sampled) * stake
        total_return = sampled["payout"].sum()

        roi = total_return / total_bet if total_bet > 0 else 0
        acc = sampled["hit"].sum() / len(sampled) if len(sampled) > 0 else 0

        roi_list.append(roi)
        acc_list.append(acc)

    # --- 7. 統計量 ---
    roi_mean = np.mean(roi_list)
    roi_ci = (np.percentile(roi_list, 2.5),
              np.percentile(roi_list, 97.5))
    acc_mean = np.mean(acc_list)

    return {
        "roi_mean": roi_mean,
        "roi_ci": roi_ci,
        "acc_mean": acc_mean,
        "roi_list": roi_list,
        "acc_list": acc_list,
        "bets_df": df_merged
    }

def add_fuku_payout(df_pred, df_payout, race_col="レースID"):
    # --- 複勝払戻だけ取り出す ---
    fuku = df_payout.query("券種 == '複勝'")[
        [race_col, "馬番", "払い戻し金額"]
    ].copy()

    # df_pred と型を合わせる
    fuku["馬番"] = fuku["馬番"].astype(float).astype(str)
    df_pred["馬番"] = df_pred["馬番"].astype(str)

    # print(fuku['馬番'].head())
    # print(df_pred['馬番'].head())

    # --- マージ ---
    merged = df_pred.merge(
        fuku,
        on=[race_col, "馬番"],
        how="left"
    )

    # --- 当たり判定と払戻 ---
    merged["複勝_hit"] = merged["払い戻し金額"].notna().astype(int)
    merged["複勝払戻"] = merged["払い戻し金額"].fillna(0)

    return merged

def eval_strategy(df, min_pred, min_odds, max_odds, top_k):
    """ある1組み合わせの購入成績を計算"""
    df_f = df.copy()

    # フィルタリング
    cond = (
        (df_f['pred_score'] >= min_pred) &
        (df_f['オッズ'] >= min_odds) &
        (df_f['オッズ'] <= max_odds)
    )
    df_f = df_f[cond]

    if df_f.empty:
        return 0, 0, 0, 0, 0

    # レース単位で top-k 購入
    df_f['rank'] = df_f.groupby('レースID')['expected_value'].rank(ascending=False, method='first')
    df_buy = df_f[df_f['rank'] <= top_k]

    if df_buy.empty:
        return 0, 0, 0, 0, 0

    # 指標計算
    n_buy = len(df_buy)
    hit_rate = df_buy['複勝_hit'].mean()
    total_return = df_buy['複勝払戻'].sum()
    total_bet = n_buy * 100  # 複勝1点100円
    roi = total_return / total_bet

    return n_buy, hit_rate, roi, total_return, total_bet


def search_best_filters(df, 
                        pred_range=[-np.inf],
                        min_odds_list=[1, 2, 3, 5, 7, 10],
                        max_odds_list=[20, 30, 50, 999],
                        top_k_list=[1, 2, 3]):
    """フィルタリング全探索"""
    results = []

    for min_pred, min_odds, max_odds, top_k in itertools.product(
        pred_range, min_odds_list, max_odds_list, top_k_list
    ):
        if min_odds >= max_odds:
            continue

        n_buy, hit_rate, roi, total_return, total_bet = eval_strategy(
            df, min_pred, min_odds, max_odds, top_k
        )

        results.append({
            'min_pred': min_pred,
            'min_odds': min_odds,
            'max_odds': max_odds,
            'top_k': top_k,
            'n_buy': n_buy,
            'hit_rate': hit_rate,
            'ROI': roi,
            'total_return': total_return,
            'total_bet': total_bet,
        })

    res_df = pd.DataFrame(results)

    # 購入件数が一定以上のものだけ（極端な少数を除外）
    res_df = res_df[res_df['n_buy'] >= 30]

    # ROI順に並べる
    res_df = res_df.sort_values('ROI', ascending=False)

    return res_df

def umatan(df):
    num_hours = len(df) - len(df['レースID'].unique())
    n_boot = 10000  # ブートストラップ試行回数
    roi_list = []
    acc_list = []
    df = select_top_with_odds(df)

    # df = df.loc[df.groupby('レースID')['pred_score'].idxmax()]

    total_return = (df['is_win'] * df['馬単']).sum()
    total_bet = num_hours * 100
    roi = total_return / total_bet

    print(f'ROI: {roi:.2%}')


set_seed(4)
# # 使用例
# # df は race データで 'オッズ', '払戻', '賭金' のカラムがあること
# df = pd.read_csv(f'./csv/tokyo_result_ranknet2_test_0.csv')
df = pd.read_csv(f'./csv/chukyo_result_ranknet_test_fuku_2.csv')
# # df_payout = pd.read_csv('./csv/tokyo_payouts_2025.csv')

# # umatan(df)

# # df = add_fuku_payout(df, df_payout)
# df = select_top_with_odds(df)

df = df.loc[df.groupby('レースID')['pred_score'].idxmax()]
# print(df['pred_score'].head(10))
# df = df.loc[df.groupby('レースID')['expected_value'].idxmax()]
# # df = (
# #     df
# #     .sort_values(['レースID', 'pred_score'], ascending=[True, False])
# #     .groupby('レースID')
# #     .head(3)
# # )

# df["ev_real"] = df["is_win"] * df["オッズ"] - 1

# print("EV相関:", df["pred_score"].corr(df["ev_real"]))
# print("Odds相関:", df["pred_score"].corr(df["オッズ"]))
# print("Top200 ROI:", df.sort_values("pred_score", ascending=False).head(200)["ev_real"].mean())

# df["scaled_score"] = df.groupby("レースID")["pred_score"].transform(
#     lambda s: (s - s.mean()) / (s.std() + 1e-9)
# )

# # df = select_top_with_odds(df)

# df["score_bin"] = pd.qcut(df["scaled_score"], q=10)  # decile bins

# summary = df.groupby("score_bin").apply(
#     lambda x: pd.Series({
#         "count": len(x),
#         "win_rate": x["is_win"].mean(),
#         "roi": x["単勝オッズ"].sum() / (len(x)*100)
#     })
# )

# print(summary)

summary = calculate_roi_from_odds_fuku(df)
print(summary)

# # df = pd.read_csv(f'./csv/tokyo_result_ranknet_test_fuku_4.csv')

# # print(df_payout['馬番'].head(10))
# # print(df['馬番'].head(10))
# # df = select_top_with_odds(df)

n_boot = 10000  # ブートストラップ試行回数
roi_list = []
acc_list = []

for _ in range(n_boot):
    # レース単位でリサンプリング（復元抽出）
    sampled = df.sample(frac=1.0, replace=True)
    
    total_bet = len(sampled) * 100
    total_return = sampled['複勝払戻'].sum()  # 的中時のみ払戻あり
    
    hit_count = sampled['複勝_hit'].sum()
    roi = total_return / total_bet
    acc = hit_count / len(sampled)
    
    roi_list.append(roi)
    acc_list.append(acc)

roi_arr = np.array(roi_list)
acc_arr = np.array(acc_list)

# 点推定
mean_roi = roi_arr.mean()
mean_acc = acc_arr.mean()

# 95%信頼区間
roi_ci = np.percentile(roi_arr, [2.5, 97.5])
acc_ci = np.percentile(acc_arr, [2.5, 97.5])

print(f"\n[top評価結果test ブートストラップ評価]")
print(f"レース数: {len(df)}")
print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

# result = bootstrap_wide_roi(df, df_payout)

# mean_acc = result["acc_mean"]
# acc_ci = (
#     np.percentile(result["acc_list"], 2.5),
#     np.percentile(result["acc_list"], 97.5)
# )

# mean_roi = result["roi_mean"]
# roi_ci = result["roi_ci"]

# top = result["bets_df"][["レースID"]].drop_duplicates()

# print(f"レース数: {len(top)}")
# print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
# print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

# print(len(df))
# # top = df.loc[df.groupby('レースID')[f'pred_score'].idxmax()]
# top = select_top_with_odds(df)

# top = top.sort_values('レースID').reset_index(drop=False)
# top = top[top['レースID'].astype(str).str[:4] == '2024']

# # is_win == 1 の位置を取得
# hit_positions = top.index[top['is_win'] == 1].to_list()

# # 当たり間隔（前回当たりから何レース空いたか）
# # 最初の当たりには前がないので無視する
# hit_intervals = [
#     hit_positions[i] - hit_positions[i - 1]
#     for i in range(1, len(hit_positions))
# ]
# print(hit_intervals)
# print('max', max(hit_intervals))



# # # 使用例
# # summary_winsor = calculate_roi_from_odds(df)
# # print(summary_winsor)

# # df = (
# #     df
# #     .sort_values(['レースID', 'pred_score'], ascending=[True, False])
# #     .groupby('レースID')
# #     .head(3)
# # )

# bins = [0,3,7,20,999]
# labels = ["〜3倍", "3〜7倍", "7〜20倍", "20倍〜"]
# top["odds_bin"] = pd.cut(top["オッズ"], bins=bins, labels=labels, right=False)

# summary = (
#     top.groupby(["odds_bin"])
#       .agg(
#           count=("オッズ", "size"),
#           hit_rate=("is_win", "mean"),
#           hit_count=("is_win", "sum"),
#           total_payout=("単勝オッズ", "sum")
#       )
# )
# summary["total_bet"] = summary["count"] * 100
# summary["ROI"] = summary["total_payout"] / summary["total_bet"]

# print(summary)

# # df2 = add_racewise_pseudo_odds(df2, score_col='pred_score', race_col='レースID', R=0.8, temperature=1.0)

# # df2['オッズ'] = np.log1p(df2['オッズ'] + 1)
# # df2['オッズ差'] =  df2['オッズ'] - df2['疑似オッズ']

# # # df2['ex'] = df2['isotonic_pred_score'] * df2['オッズ']

# # # df2 = df2[df2['人気'] >= -5]
# # print(df2[['疑似オッズ', 'オッズ']].head(30))
# # top = df2.loc[df2.groupby('レースID')['オッズ差'].idxmax()]

# # ブートストラップ
# n_boot = 10000  # ブートストラップ試行回数
# roi_list = []
# acc_list = []

# for _ in range(n_boot):
#     # レース単位でリサンプリング（復元抽出）
#     sampled = top.sample(frac=1.0, replace=True)
    
#     total_bet = len(sampled) * 100
#     total_return = sampled['単勝オッズ'].sum()  # 的中時のみ払戻あり
    
#     hit_count = sampled['is_win'].sum()
#     roi = total_return / total_bet
#     acc = hit_count / len(sampled)
    
#     roi_list.append(roi)
#     acc_list.append(acc)

# roi_arr = np.array(roi_list)
# acc_arr = np.array(acc_list)

# # 点推定
# mean_roi = roi_arr.mean()
# mean_acc = acc_arr.mean()

# # 95%信頼区間
# roi_ci = np.percentile(roi_arr, [2.5, 97.5])
# acc_ci = np.percentile(acc_arr, [2.5, 97.5])

# print(f"\n[top評価結果test ブートストラップ評価]")
# print(f"レース数: {len(top)}")
# print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
# print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

# df = pd.read_csv(f'./csv/tokyo_result_ranknet2_test_4.csv')
# print(len(df))
# df = df.loc[df.groupby('レースID')[f'pred_score'].idxmax()]

# # print(len(top))

# bins = [0,3,7,20,999]
# labels = ["〜3倍", "3〜7倍", "7〜20倍", "20倍〜"]
# df["odds_bin"] = pd.cut(df["オッズ"], bins=bins, labels=labels, right=False)

# summary = (
#     df.groupby(["odds_bin"])
#       .agg(
#           count=("オッズ", "size"),
#           hit_rate=("is_win", "mean"),
#           hit_count=("is_win", "sum"),
#           total_payout=("単勝オッズ", "sum")
#       )
# )
# summary["total_bet"] = summary["count"] * 100
# summary["ROI"] = summary["total_payout"] / summary["total_bet"]

# print(summary)

# with open(save_mapping_path, "rb") as f:
#     mapping = pickle.load(f)

# # print(mapping['1人気_1距離_1コーナー通過順'][('1.0_1400.0', (-18.0, 1000.0, 13.0))])
# print(mapping['人気_1場所'][('1.0_1400.0', (-18.0, 1.0))]['勝率'])

# df_2025 = pd.read_csv(f'./csv/tokyo_result_lgb_reg-to-rank_2025_0.csv', index_col=0)

# スコア付与用
# rank_values = [1.0, 1.0, 1.0, 1.0, 1.0]

# def assign_rank_score(group):
#     # pred_score_second降順で並べて順位をつける
#     group = group.sort_values("pred_score_second", ascending=False).reset_index(drop=True)
#     # 上位5頭にスコアを付与
#     group["rank_score"] = 0.0
#     n = min(len(rank_values), len(group))
#     group.loc[:n-1, "rank_score"] = rank_values[:n]
#     return group

# df_2025 = df_2025.groupby("レースID", group_keys=False).apply(assign_rank_score)

# df_2025['expected_value'] = df_2025['rank_score'] * df_2025['odds_diff'].clip(lower=0.001)


# df_2025 = df_2025.sort_values(
#     by=['レースID', 'expected_value'],  # 第二ソートキーを指定
#     ascending=[True, False]     # pred_scoreは降順、馬番は昇順
# ).reset_index(drop=True)

# top = df_2025.loc[df_2025.groupby('レースID')['expected_value'].idxmax()]

# # ブートストラップ
# n_boot = 10000  # ブートストラップ試行回数
# roi_list = []
# acc_list = []

# for _ in range(n_boot):
#     # レース単位でリサンプリング（復元抽出）
#     sampled = top.sample(frac=1.0, replace=True)
    
#     total_bet = len(sampled) * 100
#     total_return = sampled['単勝オッズ'].sum()  # 的中時のみ払戻あり
    
#     hit_count = sampled['is_win'].sum()
#     roi = total_return / total_bet
#     acc = hit_count / len(sampled)
    
#     roi_list.append(roi)
#     acc_list.append(acc)

# roi_arr = np.array(roi_list)
# acc_arr = np.array(acc_list)

# # 点推定
# mean_roi = roi_arr.mean()
# mean_acc = acc_arr.mean()

# # 95%信頼区間
# roi_ci = np.percentile(roi_arr, [2.5, 97.5])
# acc_ci = np.percentile(acc_arr, [2.5, 97.5])

# print(f"\n[top評価結果2025 ブートストラップ評価]")
# print(f"レース数: {len(top)}")
# print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
# print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")



# print(df['父馬'].head(20))
# print(df.columns.values)
# df = pd.read_csv('./csv/chukyo_payouts_2025.csv', index_col=0)
# print(df['レースID'].unique().tolist()[-50:])
# print(df['レースID'].unique().tolist()[:5])
# df = pd.read_csv(f'./csv/df_all_hanshin_2025.csv', index_col=0)
# print(len(df['レースID'].unique().tolist()))

# df = df[df['レースID'].astype(str).str[:4].astype(int) < 2018].reset_index(drop=True)

# # # print(df.head(5))

# df.to_csv('./csv/chukyo_payouts_2025.csv', na_rep='NaN')
# print(df['レースID'].head(5))
# print(df['レースID'].tail(5))

# Learning.scraping('./csv/hanshin_2012-2025.csv', '09', 2024, 2026)
# Learning.scraping('./csv/chukyo_2012-2024.csv', '07', 2025, 2026)
# Learning.scrape_payouts_combination('./csv/chukyo_payouts_2025.csv', '07', 2023, 2026)

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
# Learning.scraping('./csv/hukushima_2012-2024.csv', '03', 2025, 2026)
# Learning.scraping('./csv/nigata_2012-2024.csv', '04')
# Learning.scraping('./csv/nakayama_2012_2025_5.csv', '06', 2024, 2026)
# Learning.scraping('./csv/tokyo_2025.csv', '05', 2025, 2026)
# Learning.scraping('./csv/chukyo_2012-2024.csv', '07')
# Learning.scraping('./csv/kyoto_2012-2025.csv', '08', 2025, 2026)
# Learning.scraping('./csv/hanshin_2012-2025.csv', '09', 2024, 2026)
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