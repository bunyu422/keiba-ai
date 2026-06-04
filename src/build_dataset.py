"""
学習用データセット（df_all_{field}_2025_add.csv）を自動生成する。

使用方法:
    python src/build_dataset.py

前提:
    - csv/df_all_{field}_2025.csv が存在する（lightgbm_main.py の前処理済みデータ）
    - csv/{field}_payouts_2025.csv が存在する（払戻データ）

出力:
    - csv/{field}_winner_baseline.csv
    - csv/df_all_{field}_2025_add.csv
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.listwise import features
from src.listwise import model_config as cfg

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.0f}".format)

def main():
    field = 'chukyo'
    csv_path = f'./csv/df_all_{field}_2025.csv'
    payout_path = f'./csv/{field}_payouts_2025.csv'

    print(f"[1/6] Loading base CSV: {csv_path}")
    df = features.load_csv(csv_path)

    print(f"[1.1/6] Adding 厩舎 (stable/trainer) from raw CSV")
    raw_path = f'./csv/{field}_2012-2024.csv'
    raw = pd.read_csv(raw_path, index_col=0)
    raw = raw.loc[:, ~raw.columns.str.contains('^Unnamed')]
    stable_raw = raw[['レースID', '馬番', '厩舎']].copy()
    stable_raw['馬番'] = stable_raw['馬番'].astype(float)
    df['馬番'] = df['馬番'].astype(float)
    before = df.shape[0]
    df = df.merge(stable_raw, on=['レースID', '馬番'], how='left')
    assert df.shape[0] == before, f"Rows changed: {before} → {df.shape[0]}"
    df['騎手_厩舎'] = df['騎手'].astype(str) + '_' + df['厩舎'].astype(str)
    print(f"   厩舎 coverage: {df['厩舎'].notna().mean():.1%}")

    print(f"[2/6] Adding payout data")
    df_pay = pd.read_csv(payout_path)
    df_pay = df_pay.sort_values(['レースID'], ascending=[True])
    df = features.add_fuku_payout(df, df_pay)

    print(f"[3/6] Computing best/av 後3F")
    df['best後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).min(axis=1)
    df['av後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).mean(axis=1)

    print(f"[4/6] Adding race condition scores")
    df = features.add_race_condition_scores(df)

    print(f"[5/6] Building winner baseline")
    base = features.build_winner_baseline(df, field)

    print(f"[6/6] Adding past diff features + history features")
    df, cols_diff, score_cols = features.add_past_diff_features(df, base)

    # cfg を初期化（add_history_features が cfg を参照するため）
    cfg.feature_cols = []
    cfg.scale_cols = []
    cfg.feature_category = []
    cfg.field = field
    df = features.add_history_features(df)

    save_path = f"./csv/df_all_{field}_2025_add.csv"
    df.to_csv(save_path, index=True)
    print(f"Done! Saved to {save_path}")
    print(f"Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
