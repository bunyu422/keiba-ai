import copy
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# === 分割モジュールの読み込み（ファイル単位）===
from src.listwise import (
    model_config as cfg,
    models,
    dataset,
    features,
    losses,
    evaluation,
)
from src.common import splits

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

# === 2. 特徴量カラムの定義（cfg から読み取り）===
scale_cols = [c for c in cfg.scale_cols if not c.endswith(('_rank', '_rel', '_z'))]
common_cols = cfg.common_cols
feature_category = cfg.feature_category
embedding_cols = cfg.embedding_cols
context_cat_cols = cfg.context_cat_cols
context_num_cols = cfg.context_num_cols
inversion_cols = cfg.inversion_cols

# === 3. ファイルパス（cfg から読み取り）===
field = cfg.field
csv_path = f'./csv/df_all_{field}_2025_add.csv'

# === 4. データ読み込み・初期カラム設定 ===
df = features.load_csv(csv_path)
target_col = cfg.target_col
feature_cols = []
cfg.feature_cols = feature_cols  # runtime で feature_cols が変わるため同期
df['is_win'] = (df['着順'] == 1).astype(int)

if __name__ == '__main__':
    # === 6. 全体設定 ===
    save = True
    print(device)

    seed = 4  # 中京完成時は seed=4
    features.set_seed(seed)
    fold_results = []

    # === 7. データ前処理 ===
    print(df.columns.values)

    # 欠損補完、滑らか関連度ラベル作成
    df['オッズ'] = df['オッズ'].fillna(df['オッズ'].median())
    df['着順'] = df['着順'].fillna(0)
    df['smooth_rel'] = features.make_smooth_relevance_labels(df)

    # 新規特徴量（データリークなし・fold前）
    df = features.add_distance_group(df)
    df = features.add_interval_class(df)
    df = features.add_last3f_race_rank(df)
    df = features.add_weight_trend_slope(df)

    # 時系列CV（レースID順に train/val/test 分割）
    random.seed(1)
    np.random.seed(1)
    splits_list, df = splits.time_series_group_cv_3split_2025(df)

    candidate_cols = ['齢', '間隔', '父馬', '騎手', '性', '齢']

    year = 2025

    print("seed:", seed)
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits_list):
        if fold != 2:
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
        if save:
            joblib.dump(sire_mapping, f"./pickle-dict/sire_dict_{field}_fold{fold}.pkl")

        train_df, j_mapping = features.target_encoding(train_df, '騎手', target_col)

        val_df['騎手_te'] = val_df['騎手'].map(j_mapping).fillna(-1)
        test_df['騎手_te'] = test_df['騎手'].map(j_mapping).fillna(-1)
        df_2025['騎手_te'] = df_2025['騎手'].map(j_mapping).fillna(-1)
        if save:
            joblib.dump(j_mapping, f"./pickle-dict/jwin_dict_{field}_fold{fold}.pkl")

        # 新規交互作用TE（fold内で計算）
        te_group_list = [
            ['距離グループ', '父馬'],
            ['騎手', '距離'],
            ['騎手', 'フィールド'],
        ]
        new_te_cols = []
        for group_cols in te_group_list:
            train_df, val_df, test_df, df_2025, te_col, te_mapping = features.add_interaction_te_fold(
                train_df, val_df, test_df, df_2025, group_cols, target=target_col
            )
            new_te_cols.append(te_col)
            if save:
                joblib.dump(te_mapping, f"./pickle-dict/{te_col}_mapping_{field}_fold{fold}.pkl")

        # 学習に使わないカラムを feature_cols から除外
        feature_cols[:] = [col for col in df.columns if col not in ['オッズ', '払い戻し金額', "複勝_hit_max", "複勝払戻_max", "複勝払戻", "複勝_hit", '人気', '馬番', '厩舎', '騎手_厩舎', 'Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

        # TE列を feature_cols に追加（df.columns には存在しないため別途追加）
        feature_cols.extend(['父馬_te', '騎手_te'] + new_te_cols)

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

        # Embedding カラム・共通カラムを特徴量から除外 + _rank/_rel/_z の重複列を削除
        embedding_cols = feature_category

        feature_cols[:] = [col for col in feature_cols if col not in embedding_cols and col not in common_cols and not col.endswith(('_rank', '_rel', '_z'))]

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
        # DIST_HIST_SIZE = 500  ← 中京完成時は不使用
        # distortion_history = deque(maxlen=DIST_HIST_SIZE)
        patience = 10  # 中京完成時は patience=10
        best_val_loss = float('inf')
        best_roi = 0
        best_ndcg = 0
        no_improve_count = 0
        best_model_weights = None
        # mse_loss_fn = nn.MSELoss()  ← 中京完成時は不使用
        # alpha = 0
        val_records = val_df.copy()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for X, y, cat_X, context_X, context_cat_X, win_labels, odds, winner in train_loader:
                X, y, cat_X, context_X, context_cat_X, win_labels, odds, winner = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), odds[0].to(device), winner[0].to(device)
                y_sum = y.detach().cpu().numpy().sum()
                preds = model(X, cat_X, context_X, context_cat_X)
                loss = losses.combined_loss(preds, y, odds)

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
                    loss = losses.combined_loss(preds, y, gain)
                    val_loss += loss.item()

            val_records['pred_score'] = np.concatenate(box)

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping（中京完成時: 改善閾値 0.1, patience=10）
            if avg_val_loss < best_val_loss - 0.1:
                best_val_loss = avg_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
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

        # === 14. 評価（中京完成時: expected_value で top-1 選択 + ブートストラップ）===
        test_df['expected_value'] = test_df['pred_score'] * test_df['オッズ']
        top = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]

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

        print(f"\n[ex評価結果test ブートストラップ評価]")
        print(f"レース数: {len(top)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

        # === 14b. pred_score で top-1 選択 + ブートストラップ評価 ===
        top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]

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

        # === 14c. val 評価（expected_value で top-1 選択 + ブートストラップ）===
        val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
        top_val = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]

        roi_list = []
        acc_list = []
        for _ in range(n_boot):
            sampled = top_val.sample(frac=1.0, replace=True)
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

        print(f"\n[ex評価結果val ブートストラップ評価]")
        print(f"レース数: {len(top_val)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

        # === 14d. val 評価（pred_score で top-1 選択 + ブートストラップ）===
        top_val = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]

        roi_list = []
        acc_list = []
        for _ in range(n_boot):
            sampled = top_val.sample(frac=1.0, replace=True)
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

        print(f"\n[top評価結果val ブートストラップ評価]")
        print(f"レース数: {len(top_val)}")
        print(f"的中率: {mean_acc:.2%}（95%CI: {acc_ci[0]:.2%} ～ {acc_ci[1]:.2%}）")
        print(f"回収率: {mean_roi:.2%}（95%CI: {roi_ci[0]:.2%} ～ {roi_ci[1]:.2%}）")

        # === 15. 結果保存 ===
        test_df.to_csv(f'./csv/{field}_result_ranknet_test_{fold}.csv', index=False)
        val_df.to_csv(f'./csv/{field}_result_ranknet_val_{fold}.csv', index=False)
        if mean_roi > 1.0:
            with open("log.txt", "a", encoding="utf-8") as f:
                f.write(f"field:{field}\n")
                f.write(f"seed:{seed}\n")
                f.write(f"roi:{mean_roi}\n")

        if save:
            torch.save(model.state_dict(), f'./model/{field}_ranknet_{fold}.pth')

