import pandas as pd

# 小数点以下の表示桁数を増やす（省略防止）
pd.set_option('display.float_format', '{:.6f}'.format)

field = 'chukyo'
fold = 2

# test / val のCSVを読み込み
test_df = pd.read_csv(f'./csv/{field}_result_ranknet_test_{fold}.csv', encoding='utf-8')
val_df = pd.read_csv(f'./csv/{field}_result_ranknet_val_{fold}.csv', encoding='utf-8')

# test と val それぞれで同じ処理を行う
for name, df in [('test', test_df), ('val', val_df)]:
    # レースIDごとに pred_score 上位2頭を抽出
    top2 = df.groupby('レースID', group_keys=False).apply(
        lambda g: g.nlargest(2, 'pred_score')
    ).reset_index(drop=True)
    # 結果を表示
    print(f"\n=== {name} top2 ===")
    print(top2[['レースID', 'pred_score', '着順', '馬番']].head(10))
    # 総レース数と抽出行数を表示
    print(f"total races: {df['レースID'].nunique()}, total rows: {len(top2)}")
