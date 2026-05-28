"""
netkeiba スクレイピング
- JRA（中央競馬）: scraping()
- 地方競馬: scraping_local()
- 払い戻し情報: scrape_payouts_combination()
"""

import os
import time
import random
import traceback

import requests
import pandas as pd
from bs4 import BeautifulSoup


# 複数 User-Agent をランダムに使いまわしてブロックを回避
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

def _make_headers():
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "Connection": "close",
    }

def _append_csv(csv_path, df):
    """CSV に追記（なければ新規作成）"""
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, na_rep='NaN')
    else:
        df.to_csv(csv_path, mode='a', header=False, na_rep='NaN')


def scraping(csv_path, no, start, end):
    """
    JRA（中央競馬）のレース結果をスクレイピングして CSV に保存する。

    Parameters
    ----------
    csv_path : str
        保存先 CSV パス
    no : str
        開催場所コード (race_id の一部)
    start : int
        開始年
    end : int
        終了年（含まない）
    """
    df = pd.DataFrame()

    for year in range(start, end):
        for number in range(1, 6):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    race_id = f"{year}{no}{str(number).zfill(2)}{str(day).zfill(2)}{str(race_no).zfill(2)}"
                    url_race = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_list"
                    url_past = f"https://race.netkeiba.com/race/shutuba_past.html?race_id={race_id}&rf=shutuba_submenu"

                    try:
                        response_race = requests.get(url_race, headers=_make_headers())
                        response_past = requests.get(url_past, headers=_make_headers())
                        df_result = pd.read_html(response_race.content)[0]
                        df_past = pd.read_html(response_past.content)[0]
                        soup = BeautifulSoup(response_race.content, 'html.parser')

                        data1 = soup.find('div', class_='RaceData01').text
                        data2 = soup.find('div', class_='RaceData02').text
                        data3 = soup.find('tr', class_='Umatan').text
                        data4 = soup.find('h1', class_='RaceName').text

                        # 新馬戦はスキップ
                        if data2[data2.find('新馬'):data2.find('新馬')+2] == '新馬':
                            continue

                        df_row = _build_race_row(df_result, df_past, data1, data2, data3, data4, race_id, jra=True)
                        df = pd.concat([df, df_row])
                        print(url_race)
                        time.sleep(random.uniform(3, 7))

                    except Exception:
                        if race_no == 1:
                            print(f"no: {url_race}")
                            break
                        continue

    _append_csv(csv_path, df)


def scraping_local(csv_path, no, start, end):
    """
    地方競馬のレース結果をスクレイピングして CSV に保存する。

    Parameters
    ----------
    csv_path : str
        保存先 CSV パス
    no : str
        開催場所コード (race_id の一部)
    start : int
        開始年
    end : int
        終了年（含まない）
    """
    df = pd.DataFrame()

    for year in range(start, end):
        for month in range(1, 13):
            for day in range(1, 32):
                for race_no in range(1, 13):
                    race_id = f"{year}{no}{str(month).zfill(2)}{str(day).zfill(2)}{str(race_no).zfill(2)}"
                    url_race = f"https://nar.netkeiba.com/race/result.html?race_id={race_id}&rf=race_list"
                    url_past = f"https://nar.netkeiba.com/race/shutuba_past.html?race_id={race_id}&rf=shutuba_submenu"

                    try:
                        response_race = requests.get(url_race, headers=_make_headers(), timeout=(10, 30))
                        response_past = requests.get(url_past, headers=_make_headers(), timeout=(10, 30))
                        df_result = pd.read_html(response_race.content)[0]
                        df_past = pd.read_html(response_past.content)[0]
                        soup = BeautifulSoup(response_race.content, 'html.parser')

                        data1 = soup.find('div', class_='RaceData01').text
                        data2 = soup.find('div', class_='RaceData02').text
                        data3 = soup.find('tr', class_='Umatan').text
                        data4 = soup.find('div', class_='RaceName').text

                        if data2[data2.find('新馬'):data2.find('新馬')+2] == '新馬':
                            continue

                        df_row = _build_race_row(df_result, df_past, data1, data2, data3, data4, race_id, jra=False)
                        df = pd.concat([df, df_row])
                        print(url_race)
                        time.sleep(1)

                    except Exception:
                        if race_no == 1:
                            print(f"no: {url_race}")
                            break
                        traceback.print_exc()
                        continue

    _append_csv(csv_path, df)


def scrape_payouts_combination(csv_path, no, start, end):
    """
    JRA の払い戻し情報（券種・馬番・金額）をスクレイピングして CSV に保存する。

    Parameters
    ----------
    csv_path : str
        保存先 CSV パス
    no : str
        開催場所コード
    start : int
        開始年
    end : int
        終了年（含まない）
    """
    df = pd.DataFrame(columns=['レースID', '券種', '馬番', '払い戻し金額'])

    for year in range(start, end):
        for number in range(1, 6):
            for day in range(1, 13):
                for race_no in range(1, 13):
                    race_id = f"{year}{no}{str(number).zfill(2)}{str(day).zfill(2)}{str(race_no).zfill(2)}"
                    url_race = f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_list"

                    try:
                        response = requests.get(url_race, headers=_make_headers())
                        soup = BeautifulSoup(response.content, 'html.parser')

                        data2 = soup.find('div', class_='RaceData02').text
                        if data2[data2.find('新馬'):data2.find('新馬')+2] == '新馬':
                            continue

                        rows = _parse_payout_tables(soup, race_id)
                        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
                        print(f"Scraped {race_id}")
                        time.sleep(1)

                    except Exception as e:
                        print(f"Error {race_id}: {e}")
                        continue

    _append_csv(csv_path, df)


# ------------------------------------------------------------------ #
# 内部ヘルパー
# ------------------------------------------------------------------ #

def _build_race_row(df_result, df_past, data1, data2, data3, data4, race_id, jra):
    """レース結果 DataFrame に付加情報カラムを追加して返す。"""
    import re
    df_row = pd.merge(df_result, df_past, on='馬番')
    df_row['距離'] = re.findall(r'\d+', data1)[2]
    df_row['フィールド'] = data1[data1.find('/')+2: data1.find('/')+3]
    df_row['馬場'] = data1[data1.find('馬場')+3: data1.find('馬場')+4]
    df_row['出走頭数'] = data2[data2.find('頭')-2: data2.find('頭')]
    df_row['馬単'] = data3
    race_name = data4.replace('\n', '') if jra else data4.replace('\n', '')
    df_row['レース名'] = race_name
    df_row['レースID'] = race_id
    return df_row


def _parse_payout_tables(soup, race_id):
    """払い戻しテーブルから行データのリストを返す。"""
    rows = []
    payout_tables = soup.find_all('table', class_='Payout_Detail_Table')

    for table in payout_tables:
        for tr in table.find_all('tr'):
            bet_type = tr.find('th').text.strip()
            payout_td = tr.find('td', class_='Payout')
            result_td = tr.find('td', class_='Result')

            if not payout_td or not result_td:
                continue

            combos = []
            if result_td.find_all('ul'):
                for ul in result_td.find_all('ul'):
                    horses = [s.text.strip() for s in ul.find_all('span') if s.text.strip()]
                    if horses:
                        combos.append('-'.join(horses))
            else:
                combos = [s.text.strip() for s in result_td.find_all('span') if s.text.strip()]

            payout_text = payout_td.get_text(separator='|')
            payouts = [
                p.strip().replace('円', '').replace(',', '')
                for p in payout_text.split('|') if p.strip()
            ]

            for i, combo in enumerate(combos):
                rows.append({
                    'レースID': race_id,
                    '券種': bet_type,
                    '馬番': combo,
                    '払い戻し金額': payouts[i] if i < len(payouts) else '',
                })

    return rows
