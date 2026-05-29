import re
import time
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime as dt
from datetime import timedelta
import datetime
import logging

import Listwise_func
import function

from selenium.webdriver.common.by import By

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

# ---- job ワイドテスト購入 ----
def job_test(n1, n2, n3, buyList_wide):
    if buyList_wide is None:
        return

    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, n1)
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, f"{n2}R")
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, "ワイド")
    _common.wait_and_click(driver, By.PARTIAL_LINK_TEXT, "ながし")

    umaban = buyList_wide
    print(f"{n1}{n2}R {umaban}番")

    for i in umaban:
        _common.click_wide_horse(driver, i)
        time.sleep(1)

    _common.wait_and_click(driver, By.LINK_TEXT, "金額入力画面へ")
    _common.wait_and_type(driver, By.CLASS_NAME, "ui-input-text", "1")
    _common.wait_and_click(driver, By.LINK_TEXT, "セット")
    _common.wait_and_click(driver, By.LINK_TEXT, "入力終了")
    _common.wait_and_type(driver, By.ID, "sum", f"{(len(umaban)-1) * 100}")
    _common.wait_and_click(driver, By.LINK_TEXT, "投票")

    _common.wait_alert_and_accept(driver)

    _common.wait_and_click(driver, By.LINK_TEXT, "続けて通常投票")


if __name__ == "__main__":
    options = _common.create_chrome_options(headless=False)
    driver = _common.create_driver(options)

    url = "https://n.ipat.jra.go.jp/sp/"
    job1()
    job_test("水沢", 12, None, [1, 4, 3])
