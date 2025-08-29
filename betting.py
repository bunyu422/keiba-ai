from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
import schedule
import function
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime as dt
from datetime import timedelta
import logging
import datetime

def job1():
    driver.get(url)
    # driver.maximize_window()
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    driver.find_element(By.ID,"userid").send_keys("51216084")
    driver.find_element(By.ID,"password").send_keys("3156")
    driver.find_element(By.ID,"pars").send_keys("1343")
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    driver.find_element(By.CLASS_NAME, 'btnColor').click()
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'OK').click()
    except:
        pass
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    driver.find_element(By.CLASS_NAME, 'ico_regular').click()#通常投票

#["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"]
def job(n1,n2,n3):#n1は競馬場を文字列型で上のリストから選んで入力、n2は何レースを買うか
    driver.find_element(By.PARTIAL_LINK_TEXT,n1).click()#競馬場
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    driver.find_element(By.PARTIAL_LINK_TEXT,str(n2)+'R').click()#何Rか
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    driver.find_element(By.PARTIAL_LINK_TEXT,'単勝').click()
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    b=driver.find_element(By.CLASS_NAME,'selectHorse').find_elements(By.CLASS_NAME,'ui-link')
    WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    horseName=[]#馬名
    odds=[]#オッズ
    for n in range(len(b)):
        x=b[n].text.split('\n')
        horseName.append(x[1])
        if x[2]!='取消' and x[2]!='--':
            odds.append(float(x[2]))
        else:
            odds.append('--')

    buyList = None
    
    if n1 == '中京':
        buyList = function.chukyo(n3, odds)

    elif n1 == '中山':
        buyList = function.nakayama(n3, odds)

    elif n1 == '東京':
        buyList = function.tokyo(n3, odds)
    
    elif n1 == '京都':
        buyList = function.kyoto(n3, odds)

    elif n1 == '新潟':
        buyList = function.nigata(n3, odds)
    
    elif n1 == '福島':
        buyList = function.hukushima(n3, odds)
    
    elif n1 == '小倉':
        buyList = function.kokura(n3, odds)


    if buyList is None:
        print(n1+str(n2)+'R'+"は購入しない")
        driver.find_element(By.LINK_TEXT, '式別').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT, 'レース').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'競馬場名').click()#通常投票画面に返る
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
    else:
        umaban = buyList
        print(n1 + str(n2) + 'R '+str(umaban) + "番、オッズ" + str(odds[umaban-1]))
        driver.find_element(By.CLASS_NAME,'selectHorse').find_elements(By.CLASS_NAME,'ui-link')[umaban-1].click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.CLASS_NAME, 'ui-input-text').send_keys("1")
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'セット').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.PARTIAL_LINK_TEXT,'番から').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'取消').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'入力終了').click()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.ID,'sum').send_keys(str(100))
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'投票').click()
        WebDriverWait(driver, 15).until(EC.alert_is_present())
        # time.sleep(4)
        Alert(driver).accept()
        WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)
        driver.find_element(By.LINK_TEXT,'続けて通常投票').click()#通常投票画面に返る

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
            
def set_info(url_race, id_list):
    time_list = []
    dict_race = []

    # ブラウザを起動する
    with webdriver.Chrome(options=options) as driver:
        locations = []
        place_list = []
        race_id = []
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
            for k in range(sum(1 for x in dict_race if x <= 12*num and x > 12*(num-1))):
                place_list.append(i)
                race_id.append(f'{i}{str(k+1).zfill(2)}')

        race_l = []
        for i in dict_race:
            race_l.append(i % 12 + 1)
    
    return time_list, place_list, race_l, race_id


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
url="https://www.ipat.jra.go.jp/sp/"
# url = "https://race.netkeiba.com/top/race_list.html?kaisai_date=20240922"
driver.get(url)
# driver.save_screenshot("C:/Users/zazak/Downloads/sample.png")

# ログイン時間指定
# startTime="12:02"#何時にログインするか入力、9時5分は"09:05"のように入力
# schedule.every().day.at(startTime).do(job1)

# 購入レース指定
id_list = [2025060202,2025090102,2025100112]
url_race = 'https://race.netkeiba.com/top/race_list.html?kaisai_date=20250302'

timeList, keibajouList, RList, race_id = set_info(url_race)

for n in range(len(timeList)):
    schedule.every().day.at(timeList[n]).do(job,n1=keibajouList[n],n2=RList[n],n3=race_id[n])

# 終了時間指定
quitTime = "16:37"#終了時間を入力
schedule.every().day.at(quitTime).do(quitDriver)

# スケジュール開始
isWorking = True
job1()
while isWorking:
    try:
        schedule.run_pending()
        time.sleep(60)#何秒ごとに処理を実行するかを入力
    except Exception as e:
        driver.save_screenshot(f'./{ datetime.datetime.now()}.png')
        exit()
        # logging.exception("What is doing when exception happens.")
        # with open('log.txt', 'a') as o:
        #     print(logging.exception("What is doing when exception happens."), file=o)
        # job1()