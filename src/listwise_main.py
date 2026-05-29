import copy
import itertools
import pickle
import random
import warnings
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import deque
import sys
from matplotlib import pyplot as plt
import os

# === 分割モジュールの読み込み（ファイル単位）===
from src.listwise import (
    model_config as cfg,
    models,
    dataset,
    features,
    losses,
    splits,
    evaluation,
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", "{:.0f}".format)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# === 1. ハイパーパラメータ ===
n_splits = 5
num_epochs = 1000
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 2. 特徴量カラムの定義 ===
context_cat_cols = ['フィールド', '馬場']
context_num_cols = ['距離']

scale_cols = ['フィールド適性スコア', '馬場適性スコア', '距離適性スコア', '間隔', '1クラス差', '1ペース差', '父馬_te', '騎手_te',
              '1後3F_diff_rank', '1後3F_diff_rel', '1後3F_diff_z', '1タイム_diff_rank', '1タイム_diff_rel', '1タイム_diff_z', '1スピード指数_diff_rank', '1スピード指数_diff_rel', '1スピード指数_diff_z', '1馬体重_diff_rank', '1馬体重_diff_rel', '1馬体重_diff_z', '1コーナー通過順_diff_rank', '1コーナー通過順_diff_rel', '1コーナー通過順_diff_z', '1馬番_diff_rank', '1馬番_diff_rel', '1馬番_diff_z', '1斤量_diff_rank', '1斤量_diff_rel', '1斤量_diff_z', '2後3F_diff_rank', '2後3F_diff_rel', '2後3F_diff_z', '2タイム_diff_rank', '2タイム_diff_rel', '2タイム_diff_z', '2スピード指数_diff_rank', '2スピード指数_diff_rel', '2スピード指数_diff_z', '2馬体重_diff_rank', '2馬体重_diff_rel', '2馬体重_diff_z', '2コーナー通過順_diff_rank', '2コーナー通過順_diff_rel', '2コーナー通過順_diff_z', '2馬番_diff_rank', '2馬番_diff_rel', '2馬番_diff_z', '2斤量_diff_rank', '2斤量_diff_rel', '2斤量_diff_z', '3後3F_diff_rank', '3後3F_diff_rel', '3後3F_diff_z', '3タイム_diff_rank', '3タイム_diff_rel', '3タイム_diff_z', '3スピード指数_diff_rank', '3スピード指数_diff_rel', '3スピード指数_diff_z', '3馬体重_diff_rank', '3馬体重_diff_rel', '3馬体重_diff_z', '3コーナー通過順_diff_rank', '3コーナー通過順_diff_rel', '3コーナー通過順_diff_z', '3馬番_diff_rank', '3馬番_diff_rel', '3馬番_diff_z', '3斤量_diff_rank', '3斤量_diff_rel', '3斤量_diff_z', '4後3F_diff_rank', '4後3F_diff_rel', '4後3F_diff_z', '4タイム_diff_rank', '4タイム_diff_rel', '4タイム_diff_z', '4スピード指数_diff_rank', '4スピード指数_diff_rel', '4スピード指数_diff_z', '4馬体重_diff_rank', '4馬体重_diff_rel', '4馬体重_diff_z', '4コーナー通過順_diff_rank', '4コーナー通過順_diff_rel', '4コーナー通過順_diff_z', '4馬番_diff_rank', '4馬番_diff_rel', '4馬番_diff_z', '4斤量_diff_rank', '4斤量_diff_rel', '4斤量_diff_z', '5後3F_diff_rank', '5後3F_diff_rel', '5後3F_diff_z', '5タイム_diff_rank', '5タイム_diff_rel', '5タイム_diff_z', '5スピード指数_diff_rank', '5スピード指数_diff_rel', '5スピード指数_diff_z', '5馬体重_diff_rank', '5馬体重_diff_rel', '5馬体重_diff_z', '5コーナー通過順_diff_rank', '5コーナー通過順_diff_rel', '5コーナー通過順_diff_z', '5馬番_diff_rank', '5馬番_diff_rel', '5馬番_diff_z', '5斤量_diff_rank', '5斤量_diff_rel', '5斤量_diff_z', '1_past_score_rank', '1_past_score_rel', '1_past_score_z', '2_past_score_rank', '2_past_score_rel', '2_past_score_z', '3_past_score_rank', '3_past_score_rel', '3_past_score_z', '4_past_score_rank', '4_past_score_rel', '4_past_score_z', '5_past_score_rank', '5_past_score_rel', '5_past_score_z']

inversion_cols = []
common_cols = ['場所','距離','フィールド','馬場','騎手','馬番','1距離','1場所','1フィールド']

feature_category = ['父馬', '騎手', '性', '齢']

embedding_cols = feature_category

# === 3. ファイルパス ===
field = 'hanshin'
csv_path = f'./csv/df_all_{field}_2025_add.csv'

# === 4. データ読み込み・初期カラム設定 ===
df = features.load_csv(csv_path)
group_col = 'レースID'
target_col = '着順'
feature_cols = []
df['is_win'] = (df['着順'] == 1).astype(int)

# === 5. 設定を分割モジュールと同期（関数内で cfg.xxx 経由でアクセスするため）===
cfg.field = field
cfg.csv_path = csv_path
cfg.group_col = group_col
cfg.target_col = target_col
cfg.feature_cols = feature_cols
cfg.scale_cols = scale_cols
cfg.feature_category = feature_category
cfg.embedding_cols = embedding_cols
cfg.context_cat_cols = context_cat_cols
cfg.context_num_cols = context_num_cols
cfg.inversion_cols = inversion_cols
cfg.common_cols = common_cols

# ペアワイズ損失関数の候補（コメントアウトは試行済み）
pairwise_list = [
    # 'squared_hinge',
    'logistic',
    # 'hinge',
    # 'bpr',
    # 'exp',
    # 'soft_margin',
    # 'tanh',
]

# ペア重み付けモードの候補（コメントアウトは試行済み）
weight_mode_list = [
    # 'ev_i',
    # 'softmax_ev',
    # 'value_i',
    'roi',
    # 'abs_value_diff',
    # 'rank_focus',
    # 'focal_roi',
    # 'odds_aware',
]

if __name__ == '__main__':
    # === 6. 全体設定 ===
    save = False
    print(device)

    seed = 22
    features.set_seed(seed)
    fold_results = []

    # === 7. データ前処理 ===
    print(df.columns.values)

    # 欠損補完、滑らか関連度ラベル作成
    df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())
    df['着順'] = df['着順'].fillna(0)
    df['smooth_rel'] = features.make_smooth_relevance_labels(df)

    # 時系列CV（レースID順に train/val/test 分割）
    random.seed(1)
    np.random.seed(1)
    splits_list, df = splits.time_series_group_cv_3split_2025(df)

    candidate_cols = ['齢', '間隔', '父馬', '騎手', '性', '齢']

    combo_list = None
    year = 2025
    pairwise, weight_mode = 'hinge', 'softmax_ev'

    for pairwise, weight_mode in itertools.product(pairwise_list, weight_mode_list):
        print("seed:", seed)
        for fold, (train_idx, val_idx, test_idx) in enumerate(splits_list):
            # fold 0 のみ実行（1回の学習・評価）
            if fold != 0:
                continue

            train_df = df.loc[train_idx]
            val_df = df.loc[val_idx]
            test_df = df.loc[test_idx]
            year = year - 1
            print("fold_num:", len(train_df))
            print("fold_num:", len(val_df))
            print("fold_num:", len(test_df))

            # === 7.1 Target Encoding ===
            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)
            df_2025  = test_df.reset_index(drop=True)

            train_df, sire_mapping = features.target_encoding(train_df, '父馬', target_col)

            val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
            test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)
            df_2025['父馬_te'] = df_2025['父馬'].map(sire_mapping).fillna(-1)

            train_df, j_mapping = features.target_encoding(train_df, '騎手', target_col)

            val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
            test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)
            df_2025['騎手_te'] = df_2025['騎手'].map(j_mapping).fillna(-1)

            # 学習に使わないカラムを feature_cols から除外
            feature_cols[:] = [col for col in df.columns if col not in ['オッズ',"複勝_hit_max", "複勝払戻_max", "複勝払戻", "複勝_hit", '人気', '馬番', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

            # === 7.2 標準化 ===
            scaler = StandardScaler()
            train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
            val_df[scale_cols] = scaler.transform(val_df[scale_cols])
            test_df[scale_cols] = scaler.transform(test_df[scale_cols])
            df_2025[scale_cols] = scaler.transform(df_2025[scale_cols])

            if save:
                joblib.dump(scaler, f"./model/scaler_{field}_fold{fold}.pkl")
                joblib.dump(scale_cols, f"./pickle-dict/scal_cols_{field}.pkl")
                joblib.dump(feature_cols, f"./pickle-dict/feature_cols_nan_{field}.pkl")

            # === 7.3 欠損補完・カテゴリ変換 ===
            train_df, val_df, test_df, df_2025 = features.fill_nan(train_df, feature_cols), features.fill_nan(val_df, feature_cols), features.fill_nan(test_df, feature_cols), features.fill_nan(df_2025, feature_cols)

            train_df, map_dict = features.race_feature_train(train_df)
            val_df = features.race_feature_test(val_df, map_dict)
            test_df = features.race_feature_test(test_df, map_dict)
            df_2025 = features.race_feature_test(df_2025, map_dict)

            if save:
                joblib.dump(map_dict, f"./pickle-dict/category_mappings_{field}_fold{fold}.pkl")

            bad_vals = ~np.isfinite(train_df.select_dtypes(include=[np.number]))

            # Embedding カラム・共通カラムを特徴量から除外
            embedding_cols = feature_category

            feature_cols[:] = [col for col in feature_cols if col not in embedding_cols and col not in common_cols]

            if save:
                joblib.dump(feature_cols, f"./pickle-dict/feature_cols_{field}.pkl")
                joblib.dump(embedding_cols, f"./pickle-dict/embedding_cols_{field}.pkl")
                joblib.dump(context_num_cols, f"./pickle-dict/context_num_cols_{field}.pkl")
                joblib.dump(context_cat_cols, f"./pickle-dict/context_cat_cols_{field}.pkl")

            # === 8. レース単位のTensor化 ===
            X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups, win_index_train_groups = features.group_by_race(train_df)
            X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups, win_index_val_groups = features.group_by_race(val_df)
            X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups, win_index_test_groups = features.group_by_race(test_df)
            X_2025_groups, y_2025_groups, cat_2025_groups, context_2025_num_groups, context_2025_cat_groups, win_2025_groups, payout_2025_groups, win_index_2025_groups = features.group_by_race(df_2025)

            # === 9. DataLoader 構築 ===
            train_dataset = dataset.RaceDataset(X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups, win_index_train_groups)
            val_dataset = dataset.RaceDataset(X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups, win_index_val_groups)
            test_dataset = dataset.RaceDataset(X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups, win_index_test_groups)
            test_2025_dataset = dataset.RaceDataset(X_2025_groups, y_2025_groups, cat_2025_groups, context_2025_num_groups, context_2025_cat_groups, win_2025_groups, payout_2025_groups, win_index_2025_groups)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=features.set_seed(seed), generator=torch.Generator().manual_seed(seed))
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            test_2025_loader = DataLoader(test_2025_dataset, batch_size=1, shuffle=False)

            # === 10. Embedding サイズ計算・モデル構築 ===
            all_df = pd.concat([train_df, val_df, test_df, df_2025], axis=0)
            embedding_sizes = []
            for col in embedding_cols:
                n_unique = all_df[col].nunique()
                n_unique = max(n_unique, 1)
                embedding_sizes.append(n_unique + 5)

            context_embedding_sizes = []
            for col in context_cat_cols:
                n_unique = all_df[col].nunique()
                n_unique = max(n_unique, 1)
                context_embedding_sizes.append(n_unique + 5)

            emb_dim = 64
            model = models.ListNet(
                embedding_sizes=embedding_sizes,
                num_features=len(feature_cols),
                context_embedding_sizes=context_embedding_sizes,
                context_num_sizes=len(context_num_cols),
                emb_dim=emb_dim,
            )
            model.apply(features.init_weights)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

            # === 11. 学習ループ ===
            DIST_HIST_SIZE = 500
            distortion_history = deque(maxlen=DIST_HIST_SIZE)
            patience = 5
            best_val_loss = float('inf')
            best_roi = 0
            best_ndcg = 0
            no_improve_count = 0
            best_model_weights = None
            mse_loss_fn = nn.MSELoss()
            alpha = 0
            val_records = val_df.copy()
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for X, y, cat_X, context_X, context_cat_X, win_labels, odds, winner in train_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, odds, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), odds[0].to(device), winner[0].to(device)
                    y_sum = y.detach().cpu().numpy().sum()
                    preds = model(X, cat_X, context_X, context_cat_X)

                    # --- market distortion による gate 制御 ---
                    with torch.no_grad():
                        distortion = losses.market_distortion_score(preds.detach(), odds, mask=None)
                        distortion_history.append(distortion.item())
                        if len(distortion_history) > 50:
                            mu = np.mean(distortion_history)
                            sigma = np.std(distortion_history) + 1e-8
                            dist_z = (distortion.item() - mu) / sigma
                        else:
                            dist_z = 0.0

                    gate = losses.distortion_gate(dist_z)
                    gate = gate ** 2

                    # --- ランク損失（ListNet） + ROI損失の混合 ---
                    loss_rank = losses.listnet_loss(preds, y)

                    loss_roi  = losses.roi_weighted_loss(preds, odds, win_labels)
                    ROI_SCALE = loss_rank.detach().mean().item()
                    loss_roi_scaled = loss_roi * (ROI_SCALE / (loss_roi.detach().abs().mean().item() + 1e-8))
                    loss_roi_scaled = torch.tanh(loss_roi_scaled / 5.0)

                    loss = (1 - gate) * loss_rank + gate * loss_roi_scaled

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                # === 12. バリデーション ===
                model.eval()
                val_loss = 0.0
                box = []
                with torch.no_grad():
                    for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in val_loader:
                        X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                        preds = model(X, cat_X, context_X, context_cat_X)
                        box.append(preds.squeeze().cpu().numpy())

                        loss = losses.listnet_loss(preds, y)
                        val_loss += loss.item()

                val_records['pred_score'] = np.concatenate(box)

                avg_train_loss = total_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)

                print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early Stopping（val_loss が 0.01 以上改善しなければ patience 回で打ち切り）
                if avg_val_loss < best_val_loss - 0.01:
                    best_val_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_distortion_history = copy.deepcopy(distortion_history)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # ベストモデルに復元
            if best_model_weights is not None:
                model.load_state_dict(best_model_weights)

            # === 13. 各データセットの予測スコア算出 ===
            val_preds = []
            model.eval()
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in val_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    preds = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    val_preds.append(preds)

            test_scores = []
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in test_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    raw_pred = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    test_scores.append(raw_pred)

            test_2025_scores = []
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner in test_2025_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device), winner[0].to(device)
                    raw_pred = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                    test_2025_scores.append(raw_pred)

            # スコアを DataFrame に付与
            val_df = val_df.copy()
            test_df = test_df.copy()
            df_2025 = df_2025.copy()
            test_df['pred_score'] = np.concatenate(test_scores)
            val_df['pred_score'] = np.concatenate(val_preds)
            df_2025['pred_score'] = np.concatenate(test_2025_scores)

            # === 14. ゲート閾値最適化（val で最高 ROI の閾値を探索）===
            val_df = losses.compute_gate(
                val_df,
                best_distortion_history,
                no_bet_threshold=0.1
            )

            top = val_df.loc[
                val_df.groupby("レースID")["pred_score"].idxmax()
            ]

            thresholds = np.linspace(0.0, 0.5, 51)

            results = []

            for th in thresholds:
                bet = top[top["gate"] > th]
                if len(bet) < 50:
                    continue

                roi = (bet["is_win"] * bet["オッズ"]).sum() / len(bet)
                hit = bet["is_win"].mean()

                results.append({
                    "th": th,
                    "roi": roi,
                    "hit": hit,
                    "n": len(bet)
                })

            best = max(
                results,
                key=lambda r: min(r["roi"], 1.0) * np.log(r["n"])
            )
            BEST_GATE_THRESHOLD = best["th"]
            print("BEST_GATE_THRESHOLD", BEST_GATE_THRESHOLD)

            # === 15. ブートストラップ評価（val）===
            top = val_df.groupby('レースID', group_keys=False).apply(
                    lambda df_race: losses.compute_gate_for_race(df_race, best_distortion_history, no_bet_threshold=BEST_GATE_THRESHOLD)
                )
            top = top.loc[top.groupby('レースID')['pred_score'].idxmax()]

            n_boot = 10000
            roi_list = []
            acc_list = []

            for _ in range(n_boot):
                sampled = top.sample(frac=1.0, replace=True)

                total_bet = len(sampled) * 100
                total_return = sampled["単勝オッズ"].sum()

                hit_count = sampled["is_win"].sum()
                roi = total_return / total_bet
                acc = hit_count / len(sampled)

                roi_list.append(roi)
                acc_list.append(acc)

            roi_arr = np.array(roi_list)
            acc_arr = np.array(acc_list)

            mean_roi = roi_arr.mean()
            mean_acc = acc_arr.mean()

            roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            print(f"\n[top評価結果val(同率ソート無し) ブートストラップ評価]")
            print(f"レース数: {len(top)}")
            print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # === 16. ブートストラップ評価（test、同率ソート無し）===
            top = test_df.groupby('レースID', group_keys=False).apply(
                    lambda df_race: losses.compute_gate_for_race(df_race, best_distortion_history, no_bet_threshold=BEST_GATE_THRESHOLD)
                )
            top = top.loc[top.groupby('レースID')['pred_score'].idxmax()]

            n_boot = 10000
            roi_list = []
            acc_list = []

            for _ in range(n_boot):
                sampled = top.sample(frac=1.0, replace=True)

                total_bet = len(sampled) * 100
                total_return = sampled["単勝オッズ"].sum()

                hit_count = sampled["is_win"].sum()
                roi = total_return / total_bet
                acc = hit_count / len(sampled)

                roi_list.append(roi)
                acc_list.append(acc)

            roi_arr = np.array(roi_list)
            acc_arr = np.array(acc_list)

            mean_roi = roi_arr.mean()
            mean_acc = acc_arr.mean()

            roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            print(f"\n[top評価結果test(同率ソート無し) ブートストラップ評価]")
            print(f"レース数: {len(top)}")
            print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # === 17. ブートストラップ評価（test、馬番→pred_score でソート済み）===
            test_df = test_df.sort_values(
                ['レースID', 'pred_score', '馬番'],
                ascending=[True, False, True]
            )

            top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]

            n_boot = 10000
            roi_list = []
            acc_list = []

            for _ in range(n_boot):
                sampled = top.sample(frac=1.0, replace=True)

                total_bet = len(sampled) * 100
                total_return = sampled["単勝オッズ"].sum()

                hit_count = sampled["is_win"].sum()
                roi = total_return / total_bet
                acc = hit_count / len(sampled)

                roi_list.append(roi)
                acc_list.append(acc)

            roi_arr = np.array(roi_list)
            acc_arr = np.array(acc_list)

            mean_roi = roi_arr.mean()
            mean_acc = acc_arr.mean()

            roi_ci = np.percentile(roi_arr, [2.5, 97.5])
            acc_ci = np.percentile(acc_arr, [2.5, 97.5])

            print(f"\n[top評価結果test ブートストラップ評価]")
            print(f"レース数: {len(top)}")
            print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
            print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

            # === 18. 結果保存 ===
            test_df.to_csv(f'./csv/{field}_result_ranknet_test_{fold}.csv', index=False)
            if mean_roi > 1.0:
                with open("log.txt", "a", encoding="utf-8") as f:
                    f.write(f"field:{field}\n")
                    f.write(f"seed:{seed}\n")
                    f.write(f"roi:{mean_roi}\n")

            if save:
                torch.save(model.state_dict(), f'./model/{field}_ranknet_{fold}.pth')
