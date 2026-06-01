"""
データ前処理
- df_first_processing  : 今走データの基本加工
- df_big_past_processing: 前走〜5走前のデータ加工
- past_level           : 平均クラス・平均ペースの算出
- df_end_processing    : 終盤の加工（集計・不要列削除など）
- create_label_gain    : LambdaRank 用ラベル生成
- encoding             : テストデータ分離＋ターゲットエンコーディング
"""

import pickle

import numpy as np
import pandas as pd

from .config import (
    CLASS_DICT, CONDITION_MAPPING, FIELD_MAPPING, SEX_MAPPING, SPEED_DICT,
)
from .utils import (
    convert_to_float, create_unique_pickle, load_pickle,
    target_encoding, time_to_seconds,
)


# ------------------------------------------------------------------ #
# 今走の基本加工
# ------------------------------------------------------------------ #

def df_first_processing(df, name, mode='推論'):
    """
    今走データの基本加工を行う。

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        場所名（pickle ファイル名に使用）
    mode : str
        '推論' 以外なら学習モードとして pickle を新規作成する
    """
    horse_path = f"./pickle-dict/horse_jra_{name}.pkl"
    jockey_path = f"./pickle-dict/jockey_jra_{name}.pkl"

    df = df.copy()

    # 学習モードのみ: 着順に文字列が含まれる行を除外
    if mode != '推論':
        df = df[~df['着順'].isin(['除外', '取消', '未定'])]

    # 父馬・出走間隔の抽出
    src_col = '馬名_y' if df['場所'].iloc[0] <= 10 else '馬名 オッズ'
    df['父馬'] = df[src_col].str.extract(r'(\w+\s)', expand=True)
    df['間隔'] = df[src_col].str.extract(r'中\s*(\d+)\s*週', expand=True).astype(float)

    # 父馬のエンコーディング
    if mode != '推論':
        horse_mapping = create_unique_pickle(df['父馬'], horse_path)
    else:
        horse_mapping = load_pickle(horse_path)
    df['父馬'] = pd.to_numeric(df['父馬'].map(horse_mapping), errors='coerce')

    # 馬単の配当を数値化
    if '馬単' in df.columns:
        df['馬単'] = (
            df['馬単']
            .str.replace(',', '')
            .str.extract(r'(\d+円)', expand=True)[0]
            .str.extract(r'(\d+)', expand=True)[0]
        )

    # 不要列を削除
    drop_cols = ['枠_x', '枠_y', '馬名_x', '馬名_y', 'コーナー通過順', '厩舎',
                 'タイム', '騎手斤量', '着差', '後3F', '印', '馬名 オッズ',
                 '馬名  オッズ', '馬名']
    df = df.drop(drop_cols, axis=1, errors='ignore')

    # 騎手の特殊記号を除去
    for sym in ['▲', '△', '☆', '★', '◇']:
        df['騎手'] = df['騎手'].str.replace(sym, '', regex=False)

    # 性別・年齢の分割
    sex_age = df['性齢'].str.extract(r'([牝牡セ])(\d+)', expand=True)
    df['性'] = sex_age[0].map(SEX_MAPPING)
    df['齢'] = sex_age[1]

    # 馬体重・増減の分割
    weight = df['馬体重(増減)'].str.extract(r'(\d{3}).([+-0]\d*)', expand=True)
    df['馬体重'] = weight[0]
    df['体重増減'] = weight[1].str.replace(r'\+', '', regex=True)

    df['フィールド'] = df['フィールド'].map(FIELD_MAPPING)
    df['馬場'] = df['馬場'].map(CONDITION_MAPPING)

    # 騎手のエンコーディング
    if mode != '推論':
        jockey_mapping = create_unique_pickle(df['騎手'], jockey_path)
    else:
        jockey_mapping = load_pickle(jockey_path)
    df['騎手'] = df['騎手'].map(jockey_mapping)

    # 不要列を削除
    df = df.drop(['性齢', '馬体重(増減)', '馬体重', '体重増減'], axis=1)

    return convert_to_float(df)


# ------------------------------------------------------------------ #
# 前走〜5走前の加工
# ------------------------------------------------------------------ #

def df_big_past_processing(df, name, field_num):
    """
    前走〜5走前のデータを解析して特徴量化する。

    Parameters
    ----------
    df : pd.DataFrame
        df_first_processing 済みの DataFrame
    name : str
        場所名（pickle ファイル名に使用）
    field_num : int
        開催場所番号
    """
    jockey_path = f"./pickle-dict/jockey_jra_{name}.pkl"
    jockey_mapping = load_pickle(jockey_path)

    df_all = df.copy()

    for sou in range(1, 6):
        s = str(sou)
        col = '前走' if sou == 1 else f'{s}走'
        df_all = _process_one_past(df_all, df, s, col, jockey_mapping, field_num)

        if sou == 1:
            df_all = past_level(df_all)

    return df_all


def _process_one_past(df_all, df_orig, sou, col, jockey_mapping, field_num):
    """1走分の過去データを解析して df_all に結合する。"""
    import re

    # 表記揺れを正規化
    df_all[col] = (
        df_all[col]
        .astype(str)
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace('\u3000', ' ', regex=False)
        .str.replace('\xa0', ' ', regex=False)
        .str.strip()
    )

    pattern = (
        r'(\d{4}.\d{2}.\d{2})\s(\w+)\s(\d*)(.*)([ダ|芝])(\d+).*'
        r'(\d:\d{2}.\d)\s(\w)\s(\d*)頭\s(\d*)番\s(\d*)人\s(\w+)\s'
        r'(\d{2}[.]\d)(.+)(\d{2}[.]\d).\s(\d{3}).([+-0]\d*).+\((-?\d*.\d{1})'
    )
    df_split = df_all[col].str.extract(pattern, expand=True)
    df_split.columns = [
        '日付', f'{sou}場所', f'{sou}過去着順', f'{sou}レース名',
        f'{sou}フィールド', f'{sou}距離', f'{sou}タイム', f'{sou}馬場',
        f'{sou}出走馬数', f'{sou}馬番', f'{sou}人気', f'{sou}騎手',
        f'{sou}斤量', f'{sou}コーナー通過順', f'{sou}後3F',
        f'{sou}馬体重', f'{sou}体重増減', f'{sou}着差',
    ]
    df_split = df_split.drop(['日付'], axis=1)

    # コーナー通過順（4角のみ）
    df_split[f'{sou}コーナー通過順'] = (
        df_split[f'{sou}コーナー通過順']
        .str[-4:-1]
        .apply(lambda x: x if isinstance(x, str) and x.strip() else None)
        .astype(float)
        .abs()
    )

    # クラス分類
    df_split[f'{sou}クラス'] = 0
    for k, v in CLASS_DICT.items():
        df_split[f'{sou}クラス'] = df_split[f'{sou}クラス'].mask(
            df_split[f'{sou}レース名'].str.contains(k, na=False), v
        )

    # 各列のマッピング
    from src.common.config import PLACE_MAPPING
    df_split[f'{sou}場所'] = df_split[f'{sou}場所'].map(PLACE_MAPPING)
    df_split[f'{sou}フィールド'] = df_split[f'{sou}フィールド'].map(FIELD_MAPPING)
    df_split[f'{sou}馬場'] = df_split[f'{sou}馬場'].map(CONDITION_MAPPING)
    df_split[f'{sou}騎手'] = df_split[f'{sou}騎手'].map(jockey_mapping)
    df_split[f'{sou}タイム'] = df_split[f'{sou}タイム'].apply(time_to_seconds)

    # スピード指数の計算
    df_split = _calc_speed_index(df_split, sou)

    # 着差にクラス補正を加算
    df_split[f'{sou}着差'] = (
        df_split[f'{sou}着差'].astype(float)
        + df_split[f'{sou}クラス'].astype(int) * 0.5
    )

    # 上がり3Fの指数化
    df_split = _calc_last3f_index(df_split, sou)

    # 距離差・場所変化・フィールド変化
    df_split[f'{sou}距離差'] = (
        df_orig['距離'].astype(float) - df_split[f'{sou}距離'].astype(float)
    )
    df_split[f'{sou}場所変化'] = df_split[f'{sou}場所'] - field_num
    df_split[f'{sou}フィールド変化'] = df_split[f'{sou}フィールド'] - df_orig['フィールド']

    df_split = df_split.drop([f'{sou}レース名', f'{sou}騎手'], axis=1)

    return pd.concat([df_all, df_split], axis=1)


def _calc_speed_index(df_split, sou):
    """スピード指数を算出して列に追加する。"""
    for i in range(1, 26):
        for k in range(800, 3700, 10):
            for field_type, col_suffix in [(1, '1'), (2, '2')]:
                key = f'{i}{k}{col_suffix}'
                if key not in SPEED_DICT:
                    continue
                speed = SPEED_DICT[key]
                mask = (
                    (df_split[f'{sou}場所'] == i)
                    & (df_split[f'{sou}距離'].astype(float) == k)
                    & (df_split[f'{sou}フィールド'] == field_type)
                )
                if field_type == 1:
                    val = (
                        ((speed + 0.01 - df_split[f'{sou}タイム'].astype(float)) * 10)
                        * (1 / speed * 100)
                        + df_split[f'{sou}馬場'] * 10
                        + (df_split[f'{sou}斤量'].astype(float) - 55) * 2
                        + 80
                    )
                else:
                    val = (
                        ((speed + 0.01 - df_split[f'{sou}タイム'].astype(float)) * 10)
                        * (1 / speed * 100)
                        + (13 - df_split[f'{sou}馬場'] * 3)
                        + (df_split[f'{sou}斤量'].astype(float) - 55) * 2
                        + 80
                    )
                df_split.loc[mask, f'{sou}スピード指数'] = val
    return df_split


def _calc_last3f_index(df_split, sou):
    """上がり3Fを距離補正して指数化する。"""
    col = f'{sou}後3F'
    dist = df_split[f'{sou}距離'].astype(float)
    f_col = df_split[f'{sou}フィールド']

    base = df_split[col].notna() & dist.notna()
    df_split[col] = df_split[col].mask(
        base & (f_col == 1),
        df_split[col].astype(float) / (0.94 + dist / 20000)
    )
    df_split[col] = df_split[col].mask(
        base & (f_col == 2),
        df_split[col].astype(float) / (1.01 + dist / 20000)
    )
    df_split[col] = df_split[col].mask(
        base & (f_col == 3),
        df_split[col].astype(float) / (0.36 + dist * 1.5 / 100000)
    )
    return df_split


# ------------------------------------------------------------------ #
# 平均クラス・平均ペースの算出
# ------------------------------------------------------------------ #

def past_level(df_all, mode='推論'):
    """
    前走クラスと前走ペースの平均、およびその差分列を追加する。

    Parameters
    ----------
    mode : str
        '推論' 以外ならレースIDごとのグループ集計を行う
    """
    df_all = df_all.copy()
    if mode != '推論':
        df_all['平均クラス'] = df_all.groupby('レースID')['1クラス'].transform('mean')
        df_all['平均ペース'] = df_all.groupby('レースID')['1コーナー通過順'].transform('mean')
    else:
        df_all['平均クラス'] = df_all['1クラス'].mean()
        df_all['平均ペース'] = df_all['1コーナー通過順'].mean()

    df_all['1クラス差'] = df_all['平均クラス'] - df_all['1クラス']
    df_all['1ペース差'] = (df_all['平均ペース'] - df_all['1コーナー通過順']).abs()
    return df_all


# ------------------------------------------------------------------ #
# 終盤の加工
# ------------------------------------------------------------------ #

def df_end_processing(df_all, mode='推論'):
    """
    着差・スピード指数の集計列追加、不要列削除、型変換を行う。

    Parameters
    ----------
    mode : str
        '推論' 以外なら学習用の rank 列を作成する
    """
    df_all = df_all.copy()

    past_cols_diff = [f'{i}着差' for i in range(1, 6)]
    past_cols_spd = [f'{i}スピード指数' for i in range(1, 6)]
    past_cols_3f = [f'{i}後3F' for i in range(1, 6)]

    df_all['best着差'] = df_all[past_cols_diff].astype(float).min(axis=1)
    df_all['bestスピード指数'] = df_all[past_cols_spd].max(axis=1)
    df_all['av着差'] = df_all[past_cols_diff].astype(float).mean(axis=1)
    df_all['avスピード指数'] = df_all[past_cols_spd].mean(axis=1)
    df_all['best後3F'] = df_all[past_cols_3f].astype(float).min(axis=1)
    df_all['av後3F'] = df_all[past_cols_3f].astype(float).mean(axis=1)
    df_all['払い戻し金額'] = np.nan

    df_all = df_all.drop(
        ['前走', '2走', '3走', '4走', '5走', 'レース名', '勝率'],
        axis=1, errors='ignore'
    )

    # 上昇度カラムの作成
    cols = [f'{i}過去着順' for i in range(5, 0, -1)]
    df_all[cols] = df_all[cols].apply(pd.to_numeric, errors='coerce')
    null_cnt = df_all[cols].isnull().sum(axis=1) + 1
    df_all['上昇度'] = (
        (df_all['5過去着順'] - df_all['4過去着順'])
        + (df_all['4過去着順'] - df_all['3過去着順'])
        + (df_all['3過去着順'] - df_all['2過去着順'])
        + (df_all['2過去着順'] - df_all['1過去着順']) / null_cnt
    )

    df_all = df_all.replace(['', '未定', '除外', '取消', '失格', '中止'], np.nan)

    if mode != '推論':
        df_all['オッズ'] = df_all['単勝オッズ']
        df_all['単勝オッズ'] = df_all['単勝オッズ'].where(
            df_all['着順'].astype(float) == 1, 0
        ) * 100
        df_all['rank'] = df_all['単勝オッズ']

        # レース内の順序をシャッフル → レースIDでソート
        df_all = df_all.sample(frac=1).sort_values('レースID', ascending=True)

    return convert_to_float(df_all)


# ------------------------------------------------------------------ #
# LambdaRank 用ラベル生成
# ------------------------------------------------------------------ #

def create_label_gain(df_all, n_bins=18):
    """
    単勝オッズに基づいて LambdaRank 用の rank ラベルと label_gain を生成する。

    Parameters
    ----------
    df_all : pd.DataFrame
    n_bins : int
        ビン数（デフォルト 18）

    Returns
    -------
    df_all : pd.DataFrame
        rank 列が追加された DataFrame
    label_gain : list
        LambdaRank に渡す gain リスト
    """
    df_all = df_all.copy()
    df_all['rank'] = df_all['着順'].astype(float).apply(lambda r: 1 / r)
    df_all['rank'] *= df_all['出走頭数'].astype(float) / n_bins

    bins = np.linspace(0, 1, df_all['rank'].nunique())
    df_all['rank'] = np.digitize(df_all['rank'], bins, right=True) - 1

    label_gain = [np.sqrt(x) for x in range(df_all['rank'].max() + 1)]
    return df_all, label_gain


# ------------------------------------------------------------------ #
# テストデータ分離＋ターゲットエンコーディング
# ------------------------------------------------------------------ #

def encoding(df_all, field):
    """
    2020年以降のデータをテストセットとして分離し、
    父馬列にターゲットエンコーディングを適用する。

    Parameters
    ----------
    df_all : pd.DataFrame
    field : str
        場所名（pickle 保存ファイル名に使用）

    Returns
    -------
    pd.DataFrame
    """
    df_all = df_all.copy()
    SPLIT_YEAR = 202000000000

    df_test = df_all[df_all['レースID'] >= SPLIT_YEAR].copy()
    df_train = df_all[df_all['レースID'] < SPLIT_YEAR]

    mapping = df_train.groupby('父馬')['rank'].mean().to_dict()
    df_test['父馬'] = pd.to_numeric(df_test['父馬'].astype(float).map(mapping), errors='coerce')

    with open(f'./pickle-dict/sire_dict{field}.pkl', 'wb') as f:
        pickle.dump(mapping, f)

    df_train = target_encoding(df_train, '父馬', 'rank')
    return pd.concat([df_train, df_test], axis=0)
