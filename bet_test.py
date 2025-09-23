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

    wait_and_click(By.CLASS_NAME, "btnBrown")

    try:
        wait_and_click(By.LINK_TEXT, "OK")
    except:
        pass

    wait_and_click(By.CLASS_NAME, "ico_regular")  # 通常投票

# ---- job ----
def job(n1, n2, n3, buyList_wide):
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
            
        wait_and_click(By.LINK_TEXT, "金額入力画面へ")
        wait_and_type(By.CLASS_NAME, "ui-input-text", "1")
        wait_and_click(By.LINK_TEXT, "セット")
        # wait_and_click(By.PARTIAL_LINK_TEXT, "番から")
        # wait_and_click(By.LINK_TEXT, "取消")
        wait_and_click(By.LINK_TEXT, "入力終了")
        wait_and_type(By.ID, "sum", f"{(len(umaban)-1) * 100}")
        wait_and_click(By.LINK_TEXT, "投票")

        wait_alert_and_accept()  # 投票確認アラート

        wait_and_click(By.LINK_TEXT, "続けて通常投票")


# ブラウザ立ち上げ
options = Options()
# options.add_argument("--headless")#ヘッドレスの切替
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
    url="https://n.ipat.jra.go.jp/sp/"
    job1()
    job("水沢", 12, None, [1,4,3])
