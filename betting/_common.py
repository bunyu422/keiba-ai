"""
betting/ 共通Selenium操作ユーティリティ

JRA（中央）・地方の両IPATサイトで共用するブラウザ操作関数をまとめる。
各スクリプトは本モジュールを import し、driver を第一引数に渡して利用する。
"""

import json
import os
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException


def create_chrome_options(headless=True):
    """共通の ChromeOptions を生成する"""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--blink-settings=imagesEnabled=false")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-default-apps")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-features=Translate")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--hide-scrollbars")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--mute-audio")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--propagate-iph-for-testing")
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument("--log-level=3")
    options.add_argument("--disable-gcm")
    return options


def create_driver(options=None):
    """ChromeOptions を受け取り WebDriver を生成して返す"""
    if options is None:
        options = create_chrome_options()
    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(10)
    return driver


def wait_and_click(driver, by, value, timeout=15):
    """要素がクリック可能になるまで待機してクリック"""
    el = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((by, value))
    )
    el.click()
    return el


def click_wide_horse(driver, umaban):
    """ワイド選択画面で馬番をクリックする"""
    print("selectHorse count:", driver.page_source.count("selectHorse"))
    horse_links = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, ".selectHorse:not([style*='display: none']) .ui-link")
        )
    )
    horse_el = horse_links[umaban - 1]
    horse_el.click()


def wait_and_type(driver, by, value, text, timeout=15):
    """入力フィールドにテキストを入力（クリア後）"""
    el = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((by, value))
    )
    el.clear()
    el.send_keys(text)
    return el


def wait_alert_and_accept(driver, timeout=15):
    """アラートが表示されるまで待機して受け入れる"""
    WebDriverWait(driver, timeout).until(EC.alert_is_present())
    Alert(driver).accept()


def save_screenshot(driver, label=""):
    """エラー時にスクリーンショットを保存"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    driver.save_screenshot(f'./img/{timestamp}{suffix}.png')


def write_log(message):
    """ログファイルに追記"""
    from datetime import datetime
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {message}\n")


def handle_error(driver, context=""):
    """例外発生時の共通エラーハンドリング（スクショ＋ログ）"""
    save_screenshot(driver, context)
    write_log(f"[{context}] {traceback.format_exc()}")


def load_config():
    """betting/config.json からログイン情報を読み込む"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


# 主要Seleniumクラスを再エクスポート（各スクリプトが module-level で使えるように）
__all__ = [
    "create_chrome_options",
    "create_driver",
    "wait_and_click",
    "click_wide_horse",
    "wait_and_type",
    "wait_alert_and_accept",
    "save_screenshot",
    "write_log",
    "handle_error",
    "load_config",
]
