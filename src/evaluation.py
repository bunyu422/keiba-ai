import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

# 検証データで温度付きsoftmax → 期待値評価 → ROI/的中率を出力
def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()

    def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        return softmax

    for i in range(1, 21):
        T = 0.2
        softmax_T = make_softmax_with_temperature(T)
        val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
        val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
        top_by_race = val_df.groupby('レースID').apply(
            lambda df: df.sort_values('expected_value', ascending=False)
        )
        selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]
        selected = selected[selected['expected_value'] < 4]
        selected = selected[selected['オッズ'] > i]
        print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])
        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()
        hit_count = (selected['着順'] == 1).sum()
        roi = total_return / total_bet
        print(f"\n[評価結果 - Fold {i}]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

    return val_df

# レースごとにNDCG@kを計算し平均を返す
def calc_mean_ndcg(df, label_col='着順', score_col='pred_score', k=3):
    ndcgs = []
    for _, group in df.groupby("レースID"):
        max_rank = group[label_col].max()
        relevance = max_rank - group[label_col] + 1
        true_relevance = [relevance.values]
        pred_scores = [group[score_col].values]
        score = ndcg_score(true_relevance, pred_scores, k=k)
        ndcgs.append(score)
    return np.mean(ndcgs)
