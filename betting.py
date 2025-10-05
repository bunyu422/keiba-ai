import re
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
import schedule
import Listwise_func
import function
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import logging
import datetime


# ---- 共通関数 ----
from selenium.common.exceptions import StaleElementReferenceException

def wait_and_click(by, value, timeout=15):
    el = WebDriverWait(driver, timeout).until( EC.element_to_be_clickable((by, value)) ) 
    el.click() 
    return el

def click_wide_horse(umaban):
    """
    ワイド選択で馬をクリックする関数
    driver : Selenium WebDriver
    umaban : 馬番（1始まり）
    """
    print("selectHorse count:", driver.page_source.count("selectHorse"))

    # 表示中の selectHorse 内のリンクだけ取得
    horse_links = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, ".selectHorse:not([style*='display: none']) .ui-link")
        )
    )

    horse_el = horse_links[umaban - 1]
    horse_el.click()

def wait_and_type(by, value, text, timeout=15):
    el = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )
    el.clear()
    el.send_keys(text)
    return el

def wait_alert_and_accept(timeout=15):
    WebDriverWait(driver, timeout).until(EC.alert_is_present())
    Alert(driver).accept()

# ---- job1 を短く ----
def job1():
    driver.get(url)

    wait_and_type(By.ID, "userid", "51216084")
    wait_and_type(By.ID, "password", "3156")
    wait_and_type(By.ID, "pars", "1343")

    wait_and_click(By.CLASS_NAME, "btnColor")

    try:
        wait_and_click(By.LINK_TEXT, "OK")
    except:
        pass

    wait_and_click(By.CLASS_NAME, "ico_regular")  # 通常投票

# ---- job ----
def job(n1, n2, n3):
    wait_and_click(By.PARTIAL_LINK_TEXT, n1)        # 競馬場
    wait_and_click(By.PARTIAL_LINK_TEXT, f"{n2}R") # レース
    wait_and_click(By.PARTIAL_LINK_TEXT, "単勝")

    b = driver.find_element(By.CLASS_NAME, "selectHorse").find_elements(By.CLASS_NAME, "ui-link")

    horseName, odds = [], []
    for n in range(len(b)):
        x = b[n].text.split("\n")
        horseName.append(x[1])
        odds.append(float(x[2]) if x[2] not in ["取消", "--"] else "--")

    # どの会場用の関数を使うか
    buyList = None
    if n1 == "中山":
        field = "nakayama"
    elif n1 == "東京":
        field = "tokyo"
    elif n1 == "阪神":
        field = "hanshin"
    elif n1 == "京都":
        field = "kyoto"
    else:
        field = None

    # 馬を選択
    buyList, buyList_wide = Listwise_func.select_horse(n3, field, odds)

    if buyList is None:
        print(f"{n1}{n2}R は購入しない")
        wait_and_click(By.LINK_TEXT, "式別")
        wait_and_click(By.LINK_TEXT, "レース")
        wait_and_click(By.LINK_TEXT, "競馬場名")
    else:
        umaban = buyList
        print(f"{n1}{n2}R {umaban}番、オッズ {odds[umaban-1]}")

        driver.find_element(By.CLASS_NAME, "selectHorse").find_elements(By.CLASS_NAME, "ui-link")[umaban-1].click()

        wait_and_type(By.CLASS_NAME, "ui-input-text", "1")
        wait_and_click(By.LINK_TEXT, "セット")
        # wait_and_click(By.PARTIAL_LINK_TEXT, "番から")
        # wait_and_click(By.LINK_TEXT, "取消")
        wait_and_click(By.LINK_TEXT, "入力終了")
        wait_and_type(By.ID, "sum", "100")
        wait_and_click(By.LINK_TEXT, "投票")

        wait_alert_and_accept()  # 投票確認アラート

        wait_and_click(By.LINK_TEXT, "続けて通常投票")

    # ワイド投票
    if buyList_wide is None:
        pass
    else:
        wait_and_click(By.PARTIAL_LINK_TEXT, n1)        # 競馬場
        wait_and_click(By.PARTIAL_LINK_TEXT, f"{n2}R") # レース
        wait_and_click(By.PARTIAL_LINK_TEXT, "ワイド")
        wait_and_click(By.PARTIAL_LINK_TEXT, "ながし")

        umaban = buyList_wide
        print(f"{n1}{n2}R {umaban}番")

        for i in umaban:
            click_wide_horse(i)
            time.sleep(1)
            
        wait_and_click(By.CSS_SELECTOR, "金額入力画面へ")
        wait_and_type(By.CLASS_NAME, "ui-input-text", "1")
        wait_and_click(By.LINK_TEXT, "セット")
        # wait_and_click(By.PARTIAL_LINK_TEXT, "番から")
        # wait_and_click(By.LINK_TEXT, "取消")
        wait_and_click(By.LINK_TEXT, "入力終了")
        wait_and_type(By.ID, "sum", f"{(len(umaban)-1) * 100}")
        wait_and_click(By.LINK_TEXT, "投票")

        wait_alert_and_accept()  # 投票確認アラート

        wait_and_click(By.LINK_TEXT, "続けて通常投票")

def quitDriver():
    driver.quit()
    global isWorking#関数の外の変数を操作
    isWorking = False


# 引数：飛ばすレース番号(web左側の会場からカウント)
def set_time(skip_list, url_race):
    time_list = []

    # ブラウザを起動する
    with webdriver.Chrome(options=options) as driver:
        # ブラウザでアクセスする
        driver.get(url_race)

        # tableを取得(js反映)
        el=driver.find_elements(By.CLASS_NAME, "RaceList_Itemtime") #classでテーブルを指定

        for i in range(len(el)):
            if i+1 not in skip_list:
                time_list.append((dt.strptime(el[i].text, '%H:%M') - timedelta(minutes=3)).strftime("%H:%M"))
    
    print(time_list)
    return time_list
    
def set_RList(skip_list, num_place):
    race_l = []
    for i in range(12*num_place):
        if i+1 not in skip_list:
            race_l.append(i % 12 + 1)
    
    return race_l

def set_keibajo(num_list, name_list):
    keibajouList = []
    for i,k in zip(name_list, num_list):
        for v in range(k):
            keibajouList.append(i)
    return keibajouList

def set_id(id_list, skip_list):
    race_id = []
    count = 0
    for i in id_list:
        for k in range(12):
            if count*12+k+1 not in skip_list:
                race_id.append(f'{i}{str(k+1).zfill(2)}')
        count+=1
    return race_id
            
def set_info():
    time_list = []
    dict_race = []

    # ブラウザを起動する
    with webdriver.Chrome(options=options) as driver:
        locations = []
        place_list = []
        race_id = []

        # 当日の日付を YYYYMMDD 形式に
        today_str = datetime.datetime.today().strftime('%Y%m%d')

        # レース一覧ページ
        url_race = f'https://race.netkeiba.com/top/race_list.html?kaisai_date={today_str}'
        print(url_race)
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

        # ページ全体が読み込まれるのを待つ（例: RaceList_DataList が出るまで最大10秒）
        blocks = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "RaceList_DataList"))
        )

        for num, block in enumerate(blocks):
            # 各ブロック内の RaceList_DataItem を取得
            li_items = block.find_elements(By.CSS_SELECTOR, "li.RaceList_DataItem")
            if not li_items:
                continue

            # 最初の RaceList_DataItem を取得
            first_li = li_items[0]
            a_tag = first_li.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")

            # race_id を取得
            match = re.search(r"race_id=(\d+)", href)
            if match:
                base_id = match.group(1)[:-2]  # 末尾2桁を除く

            # 各ブロック内の RaceList_DataTitle を取得
            titles = block.find_elements(By.CLASS_NAME, "ItemTitle")
            
            for race_num, title in enumerate(titles, start=1):
                text = title.text.strip()  # 要素のテキストを取得
                if "新馬" not in text:
                    race_id.append(f'{base_id}{str(race_num).zfill(2)}')

        # tableを取得(js反映)
        el=driver.find_elements(By.CLASS_NAME, "ItemTitle") #classでテーブルを指定

        for num, i in enumerate(el, start=1):
            text = i.text.strip()  # 要素のテキストを取得
            if "新馬" not in text:
                # 「新馬」が含まれていない要素だけ処理
                # print("対象:", text)
                # ここに処理を書く
                dict_race.append(num)
                # tableを取得(js反映)

        el=driver.find_elements(By.CLASS_NAME, "RaceList_Itemtime") #classでテーブルを指定

        for i in range(len(el)):
            if i+1 in dict_race:
                time_list.append((dt.strptime(el[i].text, '%H:%M') - timedelta(minutes=3)).strftime("%H:%M"))

        for num, i in enumerate(locations, start=1):
            for k in range(sum(1 for x in dict_race if x <= 12*num and x > 12*(num-1))):
                place_list.append(i)

        race_l = []
        for i in dict_race:
            race_l.append((i-1) % 12 + 1)
        
    return time_list, place_list, race_l, race_id

# def job_with_retry(n1, n2, n3, retry=True):
#     try:
#         job(n1, n2, n3)  # 本来の購入処理
#     except Exception as e:
#         print(f"{n1}{n2}R でエラー発生: {e}")
#         if retry:
#             print("→ 1回だけリトライします")
#             try:
#                 job(n1, n2, n3)  # 再実行。ただし二重再帰しないよう retry=False
#             except Exception as e2:
#                 print(f"{n1}{n2}R リトライでも失敗: {e2}")
#                 driver.save_screenshot(f'./img/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_retry.png')
#                 with open('log.txt', 'a', encoding='utf-8') as f:
#                     f.write(f"[{datetime.datetime.now()}] {traceback.format_exc()}\n")
#                 job1()
#         else:
#             driver.save_screenshot(f'./img/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
#             with open('log.txt', 'a', encoding='utf-8') as f:
#                 f.write(f"[{datetime.datetime.now()}] {traceback.format_exc()}\n")

def job_with_retry(n1, n2, n3, retry=True):
    try:
        job(n1, n2, n3)  # 本来の購入処理
    except Exception as e:
        print(f"{n1}{n2}R でエラー発生: {e}")
        driver.save_screenshot(f'./img/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_retry.png')
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now()}] {traceback.format_exc()}\n")
        job1()


# ブラウザ立ち上げ
options = Options()
options.add_argument("--headless")#ヘッドレスの切替
options.add_argument("--blink-settings=imagesEnabled=false")                                 # 画像を非表示にする。
options.add_argument("--disable-background-networking")                                      # 拡張機能の更新、セーフブラウジングサービス、アップグレード検出、翻訳、UMAを含む様々なバックグラウンドネットワークサービスを無効にする。
options.add_argument("--disable-blink-features=AutomationControlled")                        # navigator.webdriver=false となる設定。確認⇒　driver.execute_script("return navigator.webdriver")
options.add_argument("--disable-default-apps")                                               # デフォルトアプリのインストールを無効にする。
options.add_argument("--disable-dev-shm-usage")                                              # ディスクのメモリスペースを使う。DockerやGcloudのメモリ対策でよく使われる。
options.add_argument("--disable-extensions")                                                 # 拡張機能をすべて無効にする。
options.add_argument("--disable-features=Translate")                                         # Chromeの翻訳を無効にする。右クリック・アドレスバーから翻訳の項目が消える。
options.add_argument("--disable-popup-blocking")                                             # ポップアップブロックを無効にする。
options.add_argument("--hide-scrollbars")                                                    # スクロールバーを隠す。
options.add_argument("--ignore-certificate-errors")                                          # SSL認証(この接続ではプライバシーが保護されません)を無効
options.add_argument("--mute-audio")                                                         # すべてのオーディオをミュートする。
options.add_argument("--no-default-browser-check")                                           # アドレスバー下に表示される「既定のブラウザとして設定」を無効にする。
options.add_argument("--propagate-iph-for-testing")                                          # Chromeに表示される青いヒント(？)を非表示にする。
options.add_argument("--start-maximized")                                                    # ウィンドウの初期サイズを最大化。--window-position, --window-sizeの2つとは併用不可
options.add_experimental_option("excludeSwitches", ["enable-logging"])
options.add_argument("--log-level=3")
options.add_argument("--disable-gcm")

driver = webdriver.Chrome(options=options)
driver.implicitly_wait(10)

if __name__ == "__main__":
    url="https://www.ipat.jra.go.jp/sp/"
    driver.get(url)

    timeList, keibajouList, RList, race_id = set_info()
    print(timeList, keibajouList, RList, race_id)

    for n in range(len(timeList)):
        schedule.every().day.at(timeList[n]).do(job_with_retry, n1=keibajouList[n], n2=RList[n], n3=race_id[n])

    # 終了時間指定
    quitTime = "16:37"#終了時間を入力
    schedule.every().day.at(quitTime).do(quitDriver)

    # スケジュール開始
    isWorking = True
    job1()
    while isWorking:
        try:
            schedule.run_pending()
            time.sleep(30)  # ポーリング間隔を30秒でも十分
        except Exception:
            driver.save_screenshot(f'./img/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            with open('log.txt', 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.datetime.now()}] {traceback.format_exc()}\n")

    # ループが抜けたら安全に終了
    driver.quit()
