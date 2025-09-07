from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
import schedule
import betting
import function
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import logging
import datetime
import Learning

# ブラウザ立ち上げ
options = Options()
# options = webdriver.FirefoxOptions()
options.add_argument("--headless")#ヘッドレスの切替
options.add_argument("--blink-settings=imagesEnabled=false")                                 # 画像を非表示にする。
options.add_argument("--disable-background-networking")                                      # 拡張機能の更新、セーフブラウジングサービス、アップグレード検出、翻訳、UMAを含む様々なバックグラウンドネットワークサービスを無効にする。
options.add_argument("--disable-blink-features=AutomationControlled")                        # navigator.webdriver=false となる設定。確認⇒　driver.execute_script("return navigator.webdriver")
options.add_argument("--disable-default-apps")                                               # デフォルトアプリのインストールを無効にする。
options.add_argument("--disable-dev-shm-usage")                                              # ディスクのメモリスペースを使う。DockerやGcloudのメモリ対策でよく使われる。
options.add_argument("--disable-extensions")                                                 # 拡張機能をすべて無効にする。
# options.add_argument("--disable-features=DownloadBubble")                                    # ダウンロードが完了したときの通知を吹き出しから下部表示(従来の挙動)にする。
# options.add_argument('--disable-features=DownloadBubbleV2')                                  # `--incognito`を使うとき、ダイアログ(名前を付けて保存)を非表示にする。
options.add_argument("--disable-features=Translate")                                         # Chromeの翻訳を無効にする。右クリック・アドレスバーから翻訳の項目が消える。
options.add_argument("--disable-popup-blocking")                                             # ポップアップブロックを無効にする。
# options.add_argument("--headless=new")                                                       # ヘッドレスモードで起動する。
options.add_argument("--hide-scrollbars")                                                    # スクロールバーを隠す。
options.add_argument("--ignore-certificate-errors")                                          # SSL認証(この接続ではプライバシーが保護されません)を無効
# options.add_argument("--incognito")                                                          # シークレットモードで起動する。
options.add_argument("--mute-audio")                                                         # すべてのオーディオをミュートする。
options.add_argument("--no-default-browser-check")                                           # アドレスバー下に表示される「既定のブラウザとして設定」を無効にする。
options.add_argument("--propagate-iph-for-testing")                                          # Chromeに表示される青いヒント(？)を非表示にする。
options.add_argument("--start-maximized")                                                    # ウィンドウの初期サイズを最大化。--window-position, --window-sizeの2つとは併用不可
# options.add_argument("--test-type=gpu")                                                      # アドレスバー下に表示される「Chrome for Testing~~」を非表示にする。
# options.add_argument("--window-position=100,100")                                            # ウィンドウの初期位置を指定する。--start-maximizedとは併用不可
# options.add_argument("--window-size=1600,1024")                                              # ウィンドウの初期サイズを設定する。--start-maximizedとは併用不可
# options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])  # Chromeは自動テスト ソフトウェア~~ ｜ コンソールに表示されるエラー　を非表示
# options.set_capability("browserVersion", "117")                                              # `--headless=new`を使うとき、コンソールに表示されるエラーを非表示にするための必須オプション

# service = Service()
# options.add_argument("--blink-settings=imagesEnabled=false")
# options.add_argument("--window-size=1920,1080")  # ウィンドウサイズを指定
# chrome_service = fs.Service(executable_path='/Users/XXXXXXXXX/Documents/Python/Driver/chromedriver')

# options.add_argument("-headless")
# driver = webdriver.Firefox(options=options)
driver = webdriver.Chrome(options=options)
driver.implicitly_wait(10)
# wait = WebDriverWait(driver, 10)
# url="https://www.ipat.jra.go.jp/sp/"
url = "https://race.netkeiba.com/top/race_list.html?kaisai_date=20240922"
# driver.get(url)

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
print(betting.set_info())

# Learning.scraping('./csv/sapporo_2012-2024.csv', '01')
# Learning.scraping('./csv/hakodate_2012-2024.csv', '02')
# Learning.scraping('./csv/hukushima_2012-2024.csv', '03')
# Learning.scraping('./csv/nigata_2012-2024.csv', '04')
# Learning.scraping('./csv/nakayama_2012-2024.csv', '06')
# Learning.scraping('./csv/chukyo_2012-2024.csv', '07')
# Learning.scraping('./csv/kyoto_2012-2024.csv', '08')
# Learning.scraping('./csv/hanshin_2012-2024.csv', '09')
# Learning.scraping('./csv/kokura_2012-2024.csv', '10')