import re
import time
import schedule
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime as dt
from datetime import timedelta
import datetime
import logging

import Listwise_func
import function
import Lgihtgbm_func as lf

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import _common

config = _common.load_config()

# ---- job1 ログイン ----
def job1():
    driver.get(url)

    _common.wait_and_type(driver, By.ID, "userid", config["userid"])
    _common.wait_and_type(driver, By.ID, "password", config["password"])
    _common.wait_and_type(driver, By.ID, "pars", config["pars"])

    _common.wait_and_click(driver, By.CLASS_NAME, "btnBrown")

    try:
        _common.wait_and_click(driver, By.LINK_TEXT, "OK")
    except:
        pass

    _common.wait_and_click(driver, By.CLASS_NAME, "ico_regular")  # 通常投票

# ---- job 購入 ----
def job(n1, n2, n3):
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, n1)        # 競馬場
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, f"{n2}R") # レース
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, "単勝")

    b = driver.find_element(By.CLASS_NAME, "selectHorse").find_elements(By.CLASS_NAME, "ui-link")

    horseName, odds = [], []
    for n in range(len(b)):
        x = b[n].text.split("\n")
        horseName.append(x[1])
        odds.append(float(x[2]) if x[2] not in ["取消", "--"] else "--")

    # どの会場用の関数を使うか
    buyList = None
    if n1 == "笠松":
        field = "kasamatu"
    elif n1 == "門別":
        field = "monbetu"
    elif n1 == "園田":
        field = "sonoda"
    elif n1 == "名古屋":
        field = "nagoya"
    elif n1 == "佐賀":
        field = "saga"
    else:
        field = None

    buyList, buyList_wide = lf.get_race_predict(n3, field, odds)

    if buyList is None:
        print(f"{n1}{n2}R は購入しない")
        _common.wait_and_click(driver, By.LINK_TEXT, "式別")
        _common.wait_and_click(driver, By.LINK_TEXT, "レース")
        _common.wait_and_click(driver, By.LINK_TEXT, "競馬場名")
    else:
        umaban = buyList
        print(f"{n1}{n2}R {umaban}番、オッズ {odds[umaban-1]}")

        driver.find_element(By.CLASS_NAME, "selectHorse").find_elements(By.CLASS_NAME, "ui-link")[umaban-1].click()

        _common.wait_and_type(driver, By.CLASS_NAME, "ui-input-text", "1")
        _common.wait_and_click(driver, By.LINK_TEXT, "セット")
        _common.wait_and_click(driver, By.LINK_TEXT, "入力終了")
        _common.wait_and_type(driver, By.ID, "sum", "100")
        _common.wait_and_click(driver, By.LINK_TEXT, "投票")

        _common.wait_alert_and_accept(driver)

        _common.wait_and_click(driver, By.LINK_TEXT, "続けて通常投票")

    # ワイド投票
    if buyList_wide is not None:
        _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, n1)
        _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, f"{n2}R")
        _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, "ワイド")
        _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, "ながし")

        umaban = buyList_wide
        print(f"{n1}{n2}R {umaban}番")

        for i in umaban:
            _common.click_wide_horse(driver, i)

        _common.wait_and_click(driver, By.CSS_SELECTOR, "金額入力画面へ")
        _common.wait_and_type(driver, By.CLASS_NAME, "ui-input-text", "1")
        _common.wait_and_click(driver, By.LINK_TEXT, "セット")
        _common.wait_and_click(driver, By.LINK_TEXT, "入力終了")
        _common.wait_and_type(driver, By.ID, "sum", f"{(len(umaban)-1) * 100}")
        _common.wait_and_click(driver, By.LINK_TEXT, "投票")

        _common.wait_alert_and_accept(driver)

        _common.wait_and_click(driver, By.LINK_TEXT, "続けて通常投票")


def quitDriver():
    driver.quit()
    global isWorking
    isWorking = False


def set_time(skip_list, url_race):
    time_list = []
    with webdriver.Chrome(options=options) as d:
        d.get(url_race)
        el = d.find_elements(By.CLASS_NAME, "RaceList_Itemtime")
        for i in range(len(el)):
            if i + 1 not in skip_list:
                time_list.append((dt.strptime(el[i].text, '%H:%M') - timedelta(minutes=3)).strftime("%H:%M"))
    print(time_list)
    return time_list


def set_RList(skip_list, num_place):
    race_l = []
    for i in range(12 * num_place):
        if i + 1 not in skip_list:
            race_l.append(i % 12 + 1)
    return race_l


def set_keibajo(num_list, name_list):
    keibajouList = []
    for i, k in zip(name_list, num_list):
        for v in range(k):
            keibajouList.append(i)
    return keibajouList


def set_id(id_list, skip_list):
    race_id = []
    count = 0
    for i in id_list:
        for k in range(12):
            if count * 12 + k + 1 not in skip_list:
                race_id.append(f'{i}{str(k + 1).zfill(2)}')
        count += 1
    return race_id


def set_info():
    time_list = []
    dict_race = []

    with webdriver.Chrome(options=options) as d:
        locations = []
        place_list = []
        race_id = []

        today_str = datetime.datetime.today().strftime('%Y%m%d')
        url_race = f'https://nar.netkeiba.com/top/race_list.html?kaisai_date={today_str}'
        print(url_race)
        d.get(url_race)

        el = d.find_elements(By.CLASS_NAME, "RaceList_DataTitle")
        for i in el:
            text = i.get_attribute("innerText")
            parts = text.split()
            location = parts[1]
            locations.append(location)
            print(location)

        blocks = WebDriverWait(d, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, "RaceList_DataList"))
        )

        for num, block in enumerate(blocks):
            li_items = block.find_elements(By.CSS_SELECTOR, "li.RaceList_DataItem")
            if not li_items:
                continue

            first_li = li_items[0]
            a_tag = first_li.find_element(By.TAG_NAME, "a")
            href = a_tag.get_attribute("href")

            match = re.search(r"race_id=(\d+)", href)
            if match:
                base_id = match.group(1)[:-2]

            try:
                elem = WebDriverWait(d, 10).until(
                    EC.element_to_be_clickable((By.XPATH, f'//a[normalize-space()="{locations[num]}"]'))
                )
                elem.click()
                print(f"{locations[num]} をクリックしました")
            except Exception as e:
                print(f"{locations[num]} のリンクが見つかりませんでした: {e}")

            titles = block.find_elements(By.CLASS_NAME, "ItemTitle")
            for race_num, title in enumerate(titles, start=1):
                text = title.text.strip()
                if "新馬" not in text:
                    race_id.append(f'{base_id}{str(race_num).zfill(2)}')
                    dict_race.append(race_num + 12 * num)

        el = d.find_elements(By.CLASS_NAME, "RaceData")
        for i in range(len(el)):
            if i + 1 in dict_race:
                text = el[i].get_attribute("innerText")
                parts = text.split()
                time_list.append((dt.strptime(parts[0], '%H:%M') - timedelta(minutes=3)).strftime("%H:%M"))

        for num, i in enumerate(locations, start=1):
            for k in range(sum(1 for x in dict_race if x <= 12 * num and x > 12 * (num - 1))):
                place_list.append(i)

        race_l = [(i - 1) % 12 + 1 for i in dict_race]

    return time_list, place_list, race_l, race_id


def job_with_retry(n1, n2, n3, retry=True):
    try:
        job(n1, n2, n3)
    except Exception as e:
        print(f"{n1}{n2}R でエラー発生: {e}")
        _common.save_screenshot(driver, "retry")
        _common.write_log(f"[{n1}{n2}R] {traceback.format_exc()}")
        job1()


if __name__ == "__main__":
    import traceback

    options = _common.create_chrome_options(headless=True)
    driver = _common.create_driver(options)

    url = "https://n.ipat.jra.go.jp/sp/"
    driver.get(url)

    timeList, keibajouList, RList, race_id = set_info()
    print(timeList, keibajouList, RList, race_id)
    print(len(timeList), len(keibajouList), len(RList), len(race_id))

    for n in range(len(timeList)):
        schedule.every().day.at(timeList[n]).do(job_with_retry, n1=keibajouList[n], n2=RList[n], n3=race_id[n])

    quitTime = "00:00"
    schedule.every().day.at(quitTime).do(quitDriver)

    isWorking = True
    job1()
    while isWorking:
        try:
            schedule.run_pending()
            time.sleep(30)
        except Exception:
            _common.save_screenshot(driver)
            _common.write_log(f"[main] {traceback.format_exc()}")

    driver.quit()
