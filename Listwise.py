import copy
import pickle
import random
import warnings
import joblib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import evaluation
from sklearn.preprocessing import StandardScaler
import Learning
import torch.nn.functional as F
import optuna.integration.lightgbm as lgb
import optuna
import lightgbm as lgbm
from sklearn.model_selection import StratifiedGroupKFold


# 行・列ともに省略せず全て表示する設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 警告を例外に変えてトレースバックを出す
# warnings.filterwarnings("error")

# === 0. ハイパーパラメータ ===
n_splits = 5
num_epochs = 1000
batch_size = 1  # 1レースずつ処理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_cols = ['距離', 'フィールド', '馬場', '場所', '出走頭数', '平均クラス', '平均ペース']

context_cat_cols = ['フィールド', '馬場', '出走頭数', '場所']
context_num_cols = ['距離', '平均クラス', '平均ペース']

numeric_diff_cols = ['1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数',
                    '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', 
                    'best着差', 'bestスピード指数', 'best後3F', 'av着差', 'avスピード指数', 'av後3F', '斤量']

scale_cols = ['1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
            '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
            '斤量', '1斤量', '2斤量', '3斤量', '4斤量', '5斤量', '父馬_te', '間隔', '1馬体重', '2馬体重', '3馬体重', '4馬体重', '5馬体重',
            '1体重増減', '2体重増減', '3体重増減', '4体重増減', '5体重増減', '1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数',
            'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度', '1クラス差', '1ペース差']

inversion_cols = ['人気', '齢', '間隔', '1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
                '1人気', '2人気', '3人気', '4人気', '5人気', 
                '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
                'best着差', 'av着差']

feature_category = ['馬番', '性', '父馬',
                    '1場所', '2場所', '3場所', '4場所', '5場所', '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド',
                    '1距離', '2距離', '3距離', '4距離', '5距離',
                    '1馬場', '2馬場', '3馬場', '4馬場', '5馬場','1コーナー通過順', '2コーナー通過順', '3コーナー通過順', '4コーナー通過順', '5コーナー通過順',
                    '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数', '1馬番', '2馬番', '3馬番', '4馬番', '5馬番']
diff_category_place = ['1場所変化', '2場所変化', '3場所変化', '4場所変化', '5場所変化']

diff_category_field = ['1フィールド変化', '2フィールド変化', '3フィールド変化', '4フィールド変化', '5フィールド変化']


embedding_cols = feature_category + diff_category_place + diff_category_field

# file_path
###########################モデルごとに変更が必要############################
field = 'ooi'
# csv_path = f'./csv/df_all_tokyo.csv'
csv_path = f'./csv/df_all_{field}.csv'
###########################################################################

df = evaluation.load_csv(csv_path)
group_col = 'レースID'
target_col = 'smooth_rel'
feature_cols = []
df['is_win'] = (df['着順'] == 1).astype(int)

def append_col(df):
    # df = add_history_features(df)

    df['best後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).min(axis=1)
    df['av後3F'] = df.loc[:, ['1後3F', '2後3F', '3後3F', '4後3F', '5後3F']].astype(float).mean(axis=1)
    # ascending=True → 値が小さいほど小さい順位
    # → 大きい数値ほど順位が大きくなる（方向性統一）
    df['best_speed_rank_num'] = df.groupby('レースID')['bestスピード指数'].rank(method='min', ascending=True)
    df['av_speed_rank_num'] = df.groupby('レースID')['avスピード指数'].rank(method='min', ascending=True)
    df['best後3F_rank_num'] = df.groupby('レースID')['best後3F'].rank(method='min', ascending=False)
    df['av後3F_rank_num'] = df.groupby('レースID')['av後3F'].rank(method='min', ascending=False)

    df['best_speed_rank_cat'] = df['best_speed_rank_num']
    df['av_speed_rank_cat'] = df['av_speed_rank_num']
    df['best後3F_rank_cat'] = df['best後3F_rank_num']
    df['av後3F_rank_cat'] = df['av後3F_rank_num']

    feature_cols.append('best後3F')
    feature_cols.append('av後3F')
    feature_cols.append('best_speed_rank_num')
    feature_cols.append('av_speed_rank_num')
    feature_cols.append('best後3F_rank_num')
    feature_cols.append('av後3F_rank_num')
    # feature_cols.append('同場所過去率')
    # feature_cols.append('同距離過去率')

    scale_cols.append('best後3F')
    scale_cols.append('av後3F')
    scale_cols.append('best_speed_rank_num')
    scale_cols.append('av_speed_rank_num')
    scale_cols.append('best後3F_rank_num')
    scale_cols.append('av後3F_rank_num')
    # scale_cols.append('同場所過去率')
    # scale_cols.append('同距離過去率')

    feature_category.append('best_speed_rank_cat')
    feature_category.append('av_speed_rank_cat')
    feature_category.append('best後3F_rank_cat')
    feature_category.append('av後3F_rank_cat')
    # feature_category.append('同距離過去数')
    # feature_category.append('同距離過去3着内数')
    # feature_category.append('同場所過去数')
    # feature_category.append('同場所過去3着内数')
    
    
    
    return df

def add_history_features(df):
    # 距離関連
    dist_cols = [f"{i}距離" for i in range(1, 6)]
    dist_rank_cols = [f"{i}過去着順" for i in range(1, 6)]

    def safe_to_num(val):
        try:
            return float(val)
        except:
            return np.nan

    def count_same_distance(row):
        return sum(row["距離"] == row[col] for col in dist_cols)

    def count_top3_same_distance(row):
        counts = [
            safe_to_num(row[dist_rank_cols[i]])
            for i in range(5) if row["距離"] == row[dist_cols[i]]
        ]
        return sum((not np.isnan(r)) and (r <= 3) for r in counts)

    df["同距離過去数"] = df.apply(count_same_distance, axis=1)
    df["同距離過去3着内数"] = df.apply(count_top3_same_distance, axis=1)
    df['同距離過去率'] = df['同距離過去3着内数'] / (df['同距離過去数'] + 1e-12)

    # 場所関連
    loc_cols = [f"{i}場所" for i in range(1, 6)]
    loc_rank_cols = [f"{i}過去着順" for i in range(1, 6)]

    def count_same_location(row):
        return sum(row["場所"] == row[col] for col in loc_cols)

    def count_top3_same_location(row):
        counts = [
            safe_to_num(row[loc_rank_cols[i]])
            for i in range(5) if row["場所"] == row[loc_cols[i]]
        ]
        return sum((not np.isnan(r)) and (r <= 3) for r in counts)

    df["同場所過去数"] = df.apply(count_same_location, axis=1)
    df["同場所過去3着内数"] = df.apply(count_top3_same_location, axis=1)
    df['同場所過去率'] = df['同場所過去3着内数'] / (df['同場所過去数'] + 1e-12)


    return df

# Nanの処理
def fill_nan(df, cols):
    # 1. NaN を -9999 で埋める
    df[cols] = df[cols].fillna(-9999)

    # 2. 休養フラグをまとめて作る
    rest_flags = {
        f"{i}休養": (df[f"{i}過去着順"] == -9999).astype(int)
        for i in range(1, 6)
    }

    # 3. 一括で追加（断片化しない）
    df = pd.concat([df, pd.DataFrame(rest_flags, index=df.index)], axis=1)

    return df

def eval_rank(df):
    # パラメータ設定
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 特徴量と補助変数
    feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', '1休養', '2休養', '3休養', '4休養', '5休養']]
    df['score'] = np.nan
    # print(feature_cols)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df['レースID'].unique())):
        # ランクをラベル化
        train_races = df['レースID'].unique()[train_idx]
        val_races = df['レースID'].unique()[val_idx]

        # データセット分割
        train_data = df[df['レースID'].isin(train_races)]
        val_data = df[df['レースID'].isin(val_races)]

        # モデル読み込み
        with open(f'./pickle-tuner/tokyo_rank_{fold}.pkl', 'rb') as f:
            model = pickle.load(f)

        # OOF予測
        df.loc[val_data.index, 'score'] = model.predict(val_data[feature_cols], num_iteration=model.best_iteration)

def race_feature(df):
    # レースごとの特徴としてまとめる（pandasでの例）
    category = feature_category + context_cat_cols
    for col in category:
        # 1. カラムをカテゴリ型に変換
        df[col] = df[col].astype('category')

        # 2. 0始まりの整数インデックスに変換
        df[col] = df[col].cat.codes
    
    for col in diff_category_place:
        df[col] = df[col].astype(str) + '->' + df['場所'].astype(str)
        # 1. カラムをカテゴリ型に変換
        df[col] = df[col].astype('category')

        # 2. 0始まりの整数インデックスに変換
        df[col] = df[col].cat.codes

    for col in diff_category_field:
        df[col] = df[col].astype(str) + '->' + df['フィールド'].astype(str)
        # 1. カラムをカテゴリ型に変換
        df[col] = df[col].astype('category')

        # 2. 0始まりの整数インデックスに変換
        df[col] = df[col].cat.codes
    
    
    return df

# スケールの方向をそろえる
def inversion(df):
    for col in inversion_cols:
        df[col] = -df[col]
    return df


# ランク予測
# eval_rank(df)

# === 1. データの前提 ===
# 1着の馬だけ 1、それ以外 0 の one-hot ターゲットを作成
# df['win_prob'] = (df['着順'] == 1).astype(float)

# rankから経験的勝率を計算する
# df['pred_rank'] = df.groupby('レースID')['score'].rank(method='first', ascending=False)

# 2. 実着順が1着（勝利）かどうかのフラグを作成（仮に '着順' カラムがあると仮定）


# 3. 予測順位ごとに勝率を集計
# 出走頭数ビン

# rank_winrate = df.groupby('pred_rank')['is_win'].mean().rename('win_prob_by_rank')
# # print(rank_winrate)

# # 4. 各行に予測順位に応じた勝率をマージ
# df = df.merge(rank_winrate, how='left', left_on='pred_rank', right_index=True)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 数値安定化
    return e_x / e_x.sum()


# print(df['win_prob'].head(30))
# df_sorted = df.sort_values(by=['レースID', 'win_prob'], ascending=[True, False])
# print(df_sorted[['レースID', 'win_prob']].head(30))


# 数値列だけ取り出す（NaN処理したい対象列）
# num_cols = df.select_dtypes(include='number').columns.drop('レースID')
# df[num_cols] = df.groupby('レースID')[num_cols].transform(lambda x: x.fillna(x.mean()))
# df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# softmax 計算用関数
def softmax_neg_rank(rankings):
    # -着順 を exp にかけることで、着順が良いほど大きな値になる
    x = -rankings
    exp_x = np.exp(x - np.max(x))  # 安定化のため最大値を引く
    return exp_x / np.sum(exp_x)

def add_relative_features(df, race_id_col='レースID'):
    """
    df: pandas.DataFrame 元データ（差分特徴を追加したいもの）
    numeric_cols: list[str] 差分を計算したい数値特徴のカラム名リスト
    race_id_col: str レースIDのカラム名
    
    戻り値: 差分特徴を追加したDataFrame
    """
    df = df.copy()
    
    # レースごとにグループ化
    grouped = df.groupby(race_id_col)
    
    for col in numeric_diff_cols:
        # レース内平均との差分
        mean_col = f'{col}_diff_mean'
        df[mean_col] = df[col] - grouped[col].transform('mean')
        
        # レース内最小値との差分
        min_col = f'{col}_diff_min'
        df[min_col] = df[col] - grouped[col].transform('min')

        feature_cols.append(mean_col)
        feature_cols.append(min_col)
    
    return df

# レースごとに softmax を適用するため、例として 'レースID' 単位で groupby
# df['win_prob'] = df.groupby('レースID')['着順'].transform(softmax_neg_rank)

# === 2. Dataset定義 ===
class RaceDataset(Dataset):
    def __init__(self, X_groups, y_groups,
                 cat_groups, context_num_groups, context_cat_groups, win_groups, payout_groups):
        self.X_groups = X_groups
        self.y_groups = y_groups
        self.win_groups = win_groups
        self.payout_groups = payout_groups
        self.cat_groups = cat_groups
        self.context_num_groups = context_num_groups
        self.context_cat_groups = context_cat_groups

    def __len__(self):
        return len(self.X_groups)

    def __getitem__(self, idx):
        return (
            self.X_groups[idx],         # [頭数, num_features]
            self.y_groups[idx],         # [頭数]
            self.cat_groups[idx],       # [頭数, num_cat_features]
            self.context_num_groups[idx],  # [頭数, num_context_num_features]
            self.context_cat_groups[idx],  # [頭数, num_context_cat_features]
            self.win_groups[idx],       # [頭数]
            self.payout_groups[idx]    # [頭数]
        )


# === 3. モデル定義 ===
class ListNet(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding_sizes = embedding_sizes                  # ← 追加
        self.context_embedding_sizes = context_embedding_sizes  # ← 追加
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in embedding_sizes
        ])

        # contextのカテゴリ変数embedding（例: フィールド、馬場）
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in context_embedding_sizes
        ])

        # contextの数値特徴 + embの次元
        context_input_dim = context_num_sizes + len(context_embedding_sizes) * emb_dim

        # context（レースごとの共通特徴）処理用のMLP
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 馬特徴の次元をcontextと合わせるプロジェクション
        horse_input_dim = num_features + len(embedding_sizes) * emb_dim
        self.horse_proj = nn.Linear(horse_input_dim, hidden_dim)

        # fcの定義（context加算型）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*3, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.rank_gate = nn.Linear(1, hidden_dim)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, 1)
        )



    def forward(self, x, cat_X, context_num, context_cat, rank_scores=None):
        # 埋め込み層処理
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)  # [頭数, emb_dim_total]

        # 馬特徴（数値 + embedding）
        horse_features = torch.cat([x, emb], dim=1)  # [頭数, num_features + emb_dim_total]

        # ===== context処理 =====
        # context embedding（1サンプル分）
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)  # [emb_dim * num_context_cat_features]

        # context 数値特徴（1サンプル分）
        context_num = context_num[0]  # [num_context_num_features]

        # 結合 → MLPでhidden_dimに変換
        context_all = torch.cat([context_num, context_emb], dim=0)  # [context_input_dim]
        context_out = self.context_fc(context_all.unsqueeze(0))  # [1, hidden_dim]

        # contextを各馬に複製
        context_expand = context_out.expand(horse_features.size(0), -1)  # [頭数, hidden_dim]

        # ===== contextを初期化ベクトルとして加算 =====
        # horse_featuresとcontext_expandの次元を合わせる必要あり
        if horse_features.size(1) != context_expand.size(1):
            # 線形変換で合わせる
            horse_features = self.horse_proj(horse_features)  # [頭数, hidden_dim]

        # ====== 順位スコア正規化（オプション） ======
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)  # 高順位ほど1に近い
            norm_rank = norm_rank.unsqueeze(1)  # [頭数, 1]

            # ====== ゲーティング加算 ======
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = horse_features + gate * context_expand
        else:
            # 従来のcontext加算
            horse_features = horse_features + context_expand

        # horse_features = horse_features + context_expand  # 要素ごと加算

        # ====== 交互作用（Hadamard product） ======
        interaction = horse_features * context_expand  # [頭数, hidden_dim]

        # ====== 順位スコア正規化ゲート後 ======
        combined = torch.cat([horse_features, context_expand, interaction], dim=1)

        # ====== 最終出力 ======
        out = self.fc(combined)  # [頭数, 1]

        # --- Residual Connection ---
        # MLPを通した特徴に、元のcombined（馬＋context）をskip接続
        out = out + self.residual_proj(combined)

        # ===== 最終出力 =====
        return out.squeeze(-1)

# === 4. Listwise loss ===
def listnet_loss(preds, labels, gain):
    preds = preds - preds.max()
    P_z = torch.softmax(preds, dim=0)
    loss = -torch.sum(labels * torch.log(P_z + 1e-12))

    return loss

def ranknet_loss(preds, labels):
    """
    preds : Tensor, shape (n_horses,)
        モデルの予測スコア
    labels : Tensor, shape (n_horses,)
        正解ラベル（小さいほど良い順位, 例: 1着=1, 2着=2 ...）
    """

    n = preds.shape[0]

    # --- 全ペア(i, j)のスコア差を計算 ---
    diff = preds.unsqueeze(0) - preds.unsqueeze(1)   # (n, n)
    pred_prob = torch.sigmoid(diff)                  # P(i > j)

    # --- 正解ラベルに基づきペアwiseラベル作成 ---
    # labelsが小さい方が上位 → iの方が強ければ1
    target = (labels.unsqueeze(0) < labels.unsqueeze(1)).float()  # (n, n)

    # --- 損失計算（同じ馬同士は無視するのでmaskする） ---
    mask = torch.ones_like(target, dtype=torch.bool)
    mask.fill_diagonal_(False)  # 対角成分(自己比較)は除外

    loss = F.binary_cross_entropy(pred_prob[mask], target[mask])

    return loss


# def make_smooth_relevance_labels(df, max_rel=3):
#     """
#     着順に応じて滑らかに relevance を作る
#     上位ほど大きく、下位も微小な値を持つ
#     """
#     def relevance(x):
#         if x == 1:
#             return max_rel
#         elif x == 2:
#             return max(max_rel - 1, 0)
#         elif x == 3:
#             return max(max_rel - 2, 0)
#         else:
#             return max_rel / x  # 下位も微小な値
#     return df['着順'].apply(relevance)

def make_smooth_relevance_labels(df, max_rel=3):
    return df['着順'].apply(lambda x: max_rel if x == 1 else 1/(x+1))



def lambdarank_loss(preds, labels):
    """
    preds  : Tensor, shape (n_horses,)
    labels : Tensor, shape (n_horses,)
    """
    n = preds.shape[0]

    # --- ペアwiseのスコア差 ---
    diff = preds.unsqueeze(0) - preds.unsqueeze(1)   # (n, n)
    pred_prob = torch.sigmoid(diff)

    # --- ペアwiseの正解 ---
    target = (labels.unsqueeze(0) > labels.unsqueeze(1)).float()

    # --- ΔNDCG用 gain を計算 ---
    gain = torch.pow(2.0, labels.float()) - 1.0
    _, rank_order = torch.sort(preds, descending=True)
    rank_position = torch.argsort(rank_order).float() + 1.0
    discount = 1.0 / torch.log2(rank_position + 1.0)
    delta_ndcg = torch.abs(
        (gain.unsqueeze(0) * discount.unsqueeze(0) - gain.unsqueeze(1) * discount.unsqueeze(1))
    )

    # --- 損失計算 ---
    mask = torch.ones_like(target, dtype=torch.bool)
    mask.fill_diagonal_(False)
    bce = F.binary_cross_entropy(pred_prob[mask], target[mask], reduction='none')
    loss = torch.mean(delta_ndcg[mask] * bce)

    return loss

def make_rank_labels(df, group_col='レースID', pos_col='着順', n_bins=18):
    """
    出走頭数を使って順位を0..n_binsに段階化する。
    rel_raw = (出走頭数 - 着順 + 1) / 出走頭数  を使い、0..1を等幅n_binsに分割。
    returns: df with new column 'rank_label' (int 0..n_bins)
    """
    df = df.copy()
    rel_series = pd.Series(index=df.index, dtype=float)

    for race_id, g in df.groupby(group_col):
        n = len(g)
        # 相対スコア: 1.0 (1着) ... 1/n (最下位)
        rel_raw = (n - g[pos_col].values + 1) / n
        rel_series.loc[g.index] = rel_raw

    # 0..1 を n_bins 等分して離散化（0..n_bins）
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize だと右端clampなので -1 調整
    labels = np.digitize(rel_series.values, bins, right=True) - 1
    labels = np.clip(labels, 0, n_bins).astype(int)

    df['rank_label'] = labels
    return df

def make_label_gain(n_bins=18, mode='sqrt'):
    """
    n_bins: label の最大値
    mode: 'linear', 'sqrt', 'exp', 'dcg' などで調整
    returns: list length n_bins+1
    """
    if mode == 'linear':
        return list(range(0, n_bins+1))
    if mode == 'sqrt':
        return [0] + [np.sqrt(i) for i in range(1, n_bins+1)]
    if mode == 'exp':
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    if mode == 'dcg':  # DCG風 (2^rel-1)
        return [0] + [2**i - 1 for i in range(1, n_bins+1)]
    # default linear
    return list(range(0, n_bins+1))


def labmdarank_lgb(train_df, val_df, test_df, feature_cols, target_col, embedding_cols, fold):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # パラメータ設定
    rate = 0.01
    seed=42
    gain_list = make_label_gain()
    group_train = train_df.groupby("レースID").size().to_list()
    group_val = val_df.groupby("レースID").size().to_list()

    lgb_train = lgb.Dataset(train_df[feature_cols], label=train_df[target_col], categorical_feature=embedding_cols, group=group_train)
    lgb_eval = lgb.Dataset(val_df[feature_cols], label=val_df[target_col], categorical_feature=embedding_cols, reference=lgb_train, group=group_val)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': rate,
        'random_state': seed,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'num_boost_round': 10000
    }
    ####################################################################################

    # '''
    # クロスバリデーションによるハイパーパラメータの探索 3fold
    tuner = lgb.LightGBMTunerCV(params,
                                lgb_train,
                                folds=GroupKFold(n_splits=3),
                                # categorical_feature = cat_list,
                                return_cvbooster=True,
                                verbose_eval=False
                                )

    # # ハイパーパラメータ探索の実行
    tuner.run()

    # # サーチしたパラメータの表示
    best_params = tuner.best_params
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))

    print(tuner.best_score)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',  # ←ここでランキング学習と指定！
        'metric': 'ndcg',   # for lambdarank
        'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
        'ndcg_eval_at': [1,3,5,10,18],  # 3連単を予測したい
        'label_gain': gain_list,
        'learning_rate': rate,
        'random_state': seed,
        'verbose_eval': 20,
        'early_stopping_round': 20,
        'num_boost_round': 10000,
        'feature_pre_filter': best_params['feature_pre_filter'],
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'num_leaves': best_params['num_leaves'],
        'feature_fraction': best_params['feature_fraction'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq':  best_params['bagging_freq'],
        'min_child_samples': best_params['min_child_samples'],
    }

    evals_result = {}
    model = lgbm.train(params,
                    lgb_train,  # トレーニングデータの指定
                    valid_names=['valid', 'train'],    # 学習経過で表示する名称
                    valid_sets=[lgb_eval, lgb_train],  # 先頭が early stopping 判定対象
                    # categorical_feature = cat_list,
                    callbacks=[lgbm.early_stopping(stopping_rounds=20, verbose=False),
                                lgbm.record_evaluation(evals_result)]
                    )

    # pklファイルとしてモデルを保存
    with open(f"./model/tokyo_lambdarank_{fold}.pickle", "wb") as mk:
        pickle.dump(model, mk)

    # テストデータの予測 (予測クラスを返す)
    val_df['pred_score'] = model.predict(val_df[feature_cols], num_iteration=model.best_iteration)
    test_df['pred_score'] = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)

    val_df.to_csv(f'./csv/tokyo_result_lambdarank_val_{fold}.csv', index=False)
    test_df.to_csv(f'./csv/tokyo_result_lambdarank_test_{fold}.csv', index=False)



# def listnet_loss(preds, labels, gain):
#     preds = preds - preds.max()
#     Pz = torch.softmax(preds, dim=0)

#     # 勝率に gain を掛けて重み付け
#     weighted_labels = labels * gain

#     loss = -torch.sum(weighted_labels * torch.log(Pz + 1e-12))
#     return loss


def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()
    
    # with open('./model/platt.pkl', 'rb') as f:
    #     platt = pickle.load(f)

    # val_df['pred_score'] = platt.predict_proba(np.array(val_df['pred_score']).reshape(-1, 1))[:, 1]

    def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))  # 安定化のために最大値を引く
            return e_x / e_x.sum()
        return softmax
    # df_sorted = val_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df_sorted[['レースID', 'pred_score']].head(30))
    
    # val_df['log_odds'] = np.log(val_df['オッズ'] + 1)
    for i in range(1, 21):
        # T = 0.1 * i  # 例：温度を0.5に設定（小さいほど尖る）
        T = 0.2
        softmax_T = make_softmax_with_temperature(T)

        val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
        val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
        # val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
        top_by_race = val_df.groupby('レースID').apply(
            lambda df: df.sort_values('expected_value', ascending=False)
        )

    #     print(top_by_race[['softmax_score', 'オッズ', 'expected_value', 'win_prob']].head(100))

        # a = 0.3
        # b = 0.7
        # val_df['expected_value'] = (val_df['softmax_score'] ** a) * (val_df['オッズ'] ** b)

        # val_df['expected_value'] = val_df['pred_score'] * val_df['log_odds']

        # 各レースごとに期待値上位3頭を取得
        # top3_ev = (
        #     val_df.sort_values(['レースID', 'pred_score'], ascending=[True, False])
        #         .groupby('レースID')
        #         .head(3)
        # )

        # # その中で pred_score 最大の馬を1頭だけ抽出
        # selected = (
        #     top3_ev.loc[top3_ev.groupby('レースID')['expected_value'].idxmax()]
        # )
        
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
        pass
    # print(selected[['softmax_score', 'log_odds', 'expected_value']].sort_values('expected_value', ascending=False).head(20))

    
    # top = top[top['expected_value'] > 100]

    # for i in range(1, 21):
    #     T = 0.2
    #     softmax_T = make_softmax_with_temperature(T)

    #     val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
    #     val_df['expected_value'] = val_df['softmax_score'] * val_df['オッズ']
    #     top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
    #     top = top[top['expected_value'] > 0.1 * i]
    #     total_bet = len(top) * 100
    #     total_return = top['単勝オッズ'].sum()
    #     hit_count = (top['着順'] == 1).sum()
    #     roi = total_return / total_bet

    #     print(f"\n[top評価結果{i}]")
    #     print(f"レース数: {len(top)}")
    #     print(f"的中数: {int(hit_count)}")
    #     print(f"的中率: {hit_count / len(top):.2%}")
    #     print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

    return val_df

# テスト
# val_df = pd.read_csv('./csv/tokyo_result_listnet_0.csv')  # または val データ専用ファイルを読み込む

# val_df_result = evaluate_model_on_val_df(val_df, model_path='./model/tokyo_listnet_0.pth', fold=0)

# def target_encording(df, column, target):
#     tem = pd.DataFrame()
#     df_tem = pd.DataFrame()
#     df_ind = pd.DataFrame()
#     dfs = [df.iloc[i:i+int(len(df.index)/5)+1, :] for i in range(0, len(df.index), int(len(df.index) / 5) + 1)]

#     for i in range(5):
#         df_tem = dfs[i].copy()
#         df_ind = dfs.copy()
#         del df_ind[i]
#         df_ind = pd.concat([dfs[0], dfs[1], dfs[2], dfs[3]], axis=0)
#         d = df_ind.groupby(column)[target].mean()
#         dict = d.to_dict()
#         df_tem[column] = pd.to_numeric(df_tem[column].map(dict), errors='coerce')
#         tem = pd.concat([tem, df_tem], axis=0)
    
#     return tem

def target_encoding(df, col, target, n_splits=5, random_state=42):
    df = df.copy()
    df[col + "_te"] = np.nan
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    for train_idx, val_idx in kf.split(df):
        train_data = df.iloc[train_idx]
        mapping = train_data.groupby(col)[target].mean().to_dict()
        
        # val にマッピング。未知カテゴリは -1 で埋める
        val_values = df.iloc[val_idx][col]  # ← loc ではなく iloc
        df.iloc[val_idx, df.columns.get_loc(col + "_te")] = val_values.map(mapping).fillna(-1)
    
    full_mapping = df.groupby(col)[target].mean().to_dict()
    return df, full_mapping

def group_by_race(df_part):
        X_groups, y_groups, win_groups, payout_groups = [], [], [], []
        cat_groups, context_num_groups, context_cat_groups = [], [], []

        for _, g in df_part.groupby(group_col):
            X = g[feature_cols].values.astype(np.float32)
            y = g[target_col].values.astype(np.float32)  # 予測対象（勝率など）

            # 勝敗ラベルと払戻（gain用）
            is_win = g["is_win"].values.astype(np.float32)          # 0 or 1
            payout = g["オッズ"].values.astype(np.float32) - 1.0    # 払戻倍率-1（gain）

            cat_X = g[embedding_cols].values.astype(np.int64)

            context_num = g[context_num_cols].iloc[0].values.astype(np.float32)
            context_cat = g[context_cat_cols].iloc[0].values.astype(np.int64)

            num_horses = len(g)
            context_num = np.tile(context_num, (num_horses, 1))
            context_cat = np.tile(context_cat, (num_horses, 1))

            X_groups.append(torch.tensor(X, dtype=torch.float32))
            y_groups.append(torch.tensor(y, dtype=torch.float32))
            win_groups.append(torch.tensor(is_win, dtype=torch.float32))
            payout_groups.append(torch.tensor(payout, dtype=torch.float32))
            cat_groups.append(torch.tensor(cat_X, dtype=torch.long))
            context_num_groups.append(torch.tensor(context_num, dtype=torch.float32))
            context_cat_groups.append(torch.tensor(context_cat, dtype=torch.long))

        return X_groups, y_groups, cat_groups, context_num_groups, context_cat_groups, win_groups, payout_groups

def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)


# 各レースごとに計算して平均をとる
def calc_mean_ndcg(df, label_col='着順', score_col='pred_score', k=None):
    ndcgs = []

    for _, group in df.groupby("レースID"):
        # relevance: 小さい着順ほど重要なので逆にする
        # 例: 1着=3pt, 2着=2pt, ...（この例は出走頭数3の場合）
        max_rank = group[label_col].max()
        relevance = max_rank - group[label_col] + 1
        
        # sklearnは shape=(1, n_samples) の形式を求める
        true_relevance = [relevance.values]
        pred_scores = [group[score_col].values]
        
        # NDCG@k で計算
        score = ndcg_score(true_relevance, pred_scores, k=k)
        ndcgs.append(score)

    return np.mean(ndcgs)

def embedding_init():
    emb = feature_category + diff_category_place + diff_category_field
    return emb

def set_seed(seed: int = 42):
    random.seed(seed)                    
    np.random.seed(seed)                 
    torch.manual_seed(seed)              
    torch.cuda.manual_seed(seed)         
    torch.cuda.manual_seed_all(seed)     

    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader の worker 初期化用
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return seed_worker

def time_series_group_cv_3split(df, group_col="レースID", n_splits=5):
    """
    時系列順（レースID昇順）に基づくリーク防止付きクロスバリデーション
    各foldで train / val / test の3分割を生成する
    """
    unique_races = np.sort(df[group_col].unique())
    n_races = len(unique_races)
    fold_size = n_races // (n_splits + 2)  # test分も含めて少し余裕をもたせる

    splits = []

    for i in range(n_splits):
        # 各foldで範囲を決める
        train_end = (i + 1) * fold_size
        val_end = train_end + fold_size
        test_end = val_end + fold_size

        if test_end > n_races:
            break  # データが足りなくなったら終了

        train_races = unique_races[:train_end]
        val_races = unique_races[train_end:val_end]
        test_races = unique_races[val_end:test_end]

        train_idx = df[df[group_col].isin(train_races)].index
        val_idx = df[df[group_col].isin(val_races)].index
        test_idx = df[df[group_col].isin(test_races)].index

        splits.append((train_idx, val_idx, test_idx))

    return splits

if __name__ == '__main__':
    print(device)
    # print(df.head(10))
    # print(torch.version.cuda)       # "12.7" が出ればOK
    # print(torch.cuda.is_available())  # True が出ればGPU使用可能
    # print(torch.cuda.get_device_name(0))  # GPU名表示
    seed = 1
    set_seed(seed)  # 先に乱数固定

    # === 5. KFold処理 ===
    
    # ラベル作成
    df['smooth_rel'] = make_smooth_relevance_labels(df)
    # df = make_rank_labels(df)

    # 出走頭数ビン
    # bins_horses = [0, 13, 16, 100]
    # labels_horses = ['small', 'medium', 'large']
    # df['num_horses_bin'] = pd.cut(df['出走頭数'], bins=bins_horses, labels=labels_horses)

    # 反転
    df = inversion(df)

    # カラム追加
    df = append_col(df)
    df = add_relative_features(df)
    

    # gkf = GroupKFold(n_splits=n_splits)
    # for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df[group_col])):
    # --- 使用例 ---
    splits = time_series_group_cv_3split(df, group_col="レースID", n_splits=5)

    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

        # trainval: test = 8 : 2（group単位）
        # trainval_df = df.iloc[train_idx]
        # test_df = df.iloc[test_idx]

        # # train:valid = 6 : 2（group単位）
        # gss = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)  # 0.75 of 8割 = 6割
        # train_idx, val_idx = next(gss.split(trainval_df, groups=trainval_df[group_col]))

        # train_df = trainval_df.iloc[train_idx]
        # val_df = trainval_df.iloc[val_idx]

        # print("fold_num:", len(train_df))

    # ---- 外側: trainval/test = 8:2 ----
    # sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # for fold, (trainval_idx, test_idx) in enumerate(
    #     sgkf.split(df, y=df["場所"], groups=df[group_col])
    # ):
    #     # if fold == 0:
    #     #     continue
    #     trainval_df = df.iloc[trainval_idx]
    #     test_df = df.iloc[test_idx]

    #     # ---- 内側: train/val = 6:2 (trainvalの中で) ----
    #     sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    #     inner_train_idx, val_idx = next(
    #         sgkf_inner.split(trainval_df, y=trainval_df["場所"], groups=trainval_df[group_col])
    #     )

    #     train_df = trainval_df.iloc[inner_train_idx]
    #     val_df = trainval_df.iloc[val_idx]

        # # 予測順位ごとの勝率
        # win_stats = train_df.groupby('pred_rank').apply(
        #     lambda x: (x['着順'] == 1).sum() / max(len(x), 1)
        # ).reset_index(name='win_prob')

        # # train_df にマージ
        # train_df = train_df.merge(win_stats, on='pred_rank', how='left')

        # # val_df にマージ
        # val_df = val_df.merge(win_stats, on='pred_rank', how='left')

        # # test_df にマージ
        # test_df = test_df.merge(win_stats, on='pred_rank', how='left')

        # # 条件付き統計（出走頭数bin × 予想順位）
        # group_cols = ['num_horses_bin', 'pred_rank']
        # win_stats = train_df.groupby(group_cols).apply(
        #     lambda x: (x['着順'] == 1).sum() / max(len(x), 1)
        # ).reset_index(name='win_prob')


        # # === dfに勝率をマージ ===
        # train_df = train_df.merge(
        #     win_stats,
        #     how='left',
        #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
        # )

        # # === dfに勝率をマージ ===
        # val_df = val_df.merge(
        #     win_stats,
        #     how='left',
        #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
        # )

        # # === dfに勝率をマージ ===
        # test_df = test_df.merge(
        #     win_stats,
        #     how='left',
        #     on=['num_horses_bin', 'pred_rank']  # 複合キーでマージ
        # )

        # # 5. 最終的な win_prob を追加
        # train_df['win_prob'] = train_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化
        # val_df['win_prob'] = val_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化
        # test_df['win_prob'] = test_df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化

        # === 4. 特徴量エンコーディング ===

        feature_cols = [col for col in df.columns if col not in ['Unnamed: 0', 'レースID', 'rank_label', '着順', 'rank', 'smooth_rel', 'pred_rank', 'num_horses_bin', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]

        # === 6. 特徴量エンコーディング ===
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        
        train_df, sire_mapping = target_encoding(train_df, '父馬', target_col)
        with open(f'./pickle-dict/sire_dict_{field}_fold{fold}.pkl', "wb") as dd:
            pickle.dump(sire_mapping, dd)

        # val/test は train 全体の mapping を使う
        val_df['父馬_te'] = val_df['父馬'].map(sire_mapping).fillna(-1)
        test_df['父馬_te'] = test_df['父馬'].map(sire_mapping).fillna(-1)

        # zero_var = train_df[scale_cols].std()[train_df[scale_cols].std() == 0]
        # print("分散ゼロの列:", zero_var.index.tolist())

        # print("train shape:", train_df[scale_cols].shape)
        # print("val shape:", val_df[scale_cols].shape)
        # print("test shape:", test_df[scale_cols].shape)

        # print("NaN 含有数:\n", train_df[scale_cols].isna().sum())
        # print("有効サンプル数:", train_df[scale_cols].notna().sum())

        # === 7. 特徴量スケーリング ===
        scaler = StandardScaler()
        train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
        val_df[scale_cols] = scaler.transform(val_df[scale_cols])
        test_df[scale_cols] = scaler.transform(test_df[scale_cols])

        # === 0. データの前処理 ===
        # Nanの処理
        train_df, val_df, test_df = fill_nan(train_df, feature_cols), fill_nan(val_df, feature_cols), fill_nan(test_df, feature_cols)
        # カテゴリ変換
        train_df, val_df, test_df = race_feature(train_df), race_feature(val_df), race_feature(test_df)

        bad_vals = ~np.isfinite(train_df.select_dtypes(include=[np.number]))
        # print(bad_vals.sum())           # 各列ごとの個数

        # === 3. ランキング学習 ===
        # embedding_cols = feature_category + diff_category_place + diff_category_field
        # labmdarank_lgb(train_df, val_df, test_df, feature_cols, target_col, embedding_cols, fold)
        # continue

        # === 1. データの前提 ===
        embedding_cols = feature_category + diff_category_place + diff_category_field

        feature_cols = [col for col in feature_cols if col not in embedding_cols and col not in common_cols]
        
        feature_cols = joblib.load("./pickle-dict/feature_cols.pkl")
        embedding_cols = joblib.load("./pickle-dict/embedding_cols.pkl")
        context_num_cols = joblib.load("./pickle-dict/context_num_cols.pkl")
        context_cat_cols = joblib.load("./pickle-dict/context_cat_cols.pkl")

        # joblib.dump(feature_cols, "./pickle-dict/feature_cols.pkl")
        # joblib.dump(embedding_cols, "./pickle-dict/embedding_cols.pkl")
        # joblib.dump(context_num_cols, "./pickle-dict/context_num_cols.pkl")
        # joblib.dump(context_cat_cols, "./pickle-dict/context_cat_cols.pkl")

        # print(f"feature_cols: {feature_cols}")
        # print(f"embedding_cols: {embedding_cols}")
        # print(f"context_num_cols: {context_num_cols}")
        # print(f"context_cat_cols: {context_cat_cols}")
        

        X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups, = group_by_race(train_df)
        X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups = group_by_race(val_df)
        X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups = group_by_race(test_df)

        train_dataset = RaceDataset(X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups, win_train_groups, payout_train_groups)
        val_dataset = RaceDataset(X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups, win_val_groups, payout_val_groups)
        test_dataset = RaceDataset(X_test_groups, y_test_groups, cat_test_groups, context_test_num_groups, context_test_cat_groups, win_test_groups, payout_test_groups)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,worker_init_fn=set_seed(seed), generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        all_df = pd.concat([train_df, val_df, test_df], axis=0)
        embedding_sizes = []
        for col in embedding_cols:
            n_unique = all_df[col].nunique()
            n_unique = max(n_unique, 1)  # 定数列や全NaN列でも最低1
            embedding_sizes.append(n_unique + 5)  # 余裕分 +5

        context_embedding_sizes = []
        for col in context_cat_cols:
            n_unique = all_df[col].nunique()
            n_unique = max(n_unique, 1)
            context_embedding_sizes.append(n_unique + 5)

        # embedding_sizes = [train_df[col].nunique() + 5 for col in embedding_cols]  # 各カテゴリ列のクラス数
        # context_embedding_sizes = [train_df[col].nunique() + 5 for col in context_cat_cols]  # 各カテゴリ列のクラス数

        # モデル
        emb_dim = 64  
        model = ListNet(embedding_sizes=embedding_sizes, num_features=len(feature_cols), context_embedding_sizes=context_embedding_sizes, context_num_sizes=len(context_num_cols), emb_dim=emb_dim)
        model.apply(init_weights)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        # ハイパーパラメータ
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs
        # )
        
        patience = 10  # 何エポック改善がなければ終了するか
        best_val_loss = float('inf')
        best_ndcg = 0
        no_improve_count = 0
        best_model_weights = None
        mse_loss_fn = nn.MSELoss()
        alpha = 0  # MSE の比率
        val_records = val_df.copy()
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for X, y, cat_X, context_X, context_cat_X, win_labels, gain in train_loader:
                X, y, cat_X, context_X, context_cat_X, win_labels, gain = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device)
                y_sum = y.detach().cpu().numpy().sum()
                # ランク損失
                preds = model(X, cat_X, context_X, context_cat_X)
                # loss = listnet_loss(preds, y, gain)
                loss = lambdarank_loss(preds, y)

                # 回帰損失（勝率ラベルとの直接比較）
                # prob_preds = torch.softmax(preds, dim=0)
                # reg_loss = mse_loss_fn(prob_preds.squeeze(), y)
                # # print(loss.item(), reg_loss.item())

                # loss = loss + alpha * reg_loss

                # 勾配計算
                optimizer.zero_grad()
                loss.backward()

                # 勾配確認コード（例）
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
                #     else:
                #         print(f"{name}: grad is None")


                optimizer.step()
                # scheduler.step()  # ← ここでLR更新
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            box = []
            with torch.no_grad():
                for X, y, cat_X, context_X, context_cat_X, win_labels, gain in val_loader:
                    X, y, cat_X, context_X, context_cat_X, win_labels, gain = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device)
                    preds = model(X, cat_X, context_X, context_cat_X)
                    box.append(preds.squeeze().cpu().numpy())
                    # loss = listnet_loss(preds, y, gain)
                    loss = lambdarank_loss(preds, y)
                    # 回帰損失（勝率ラベルとの直接比較）
                    # prob_preds = torch.softmax(preds, dim=0)
                    # reg_loss = mse_loss_fn(prob_preds.squeeze(), y)
                    # loss = loss + alpha * reg_loss
                    val_loss += loss.item()
            
            val_records['pred_score'] = np.concatenate(box)
            ndcg = calc_mean_ndcg(val_records)
            print(f"Epoch {epoch+1}: NDCG: {ndcg:.4f}")

            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Early Stopping 判定
            if ndcg > best_ndcg + 1e-3:
                best_ndcg = ndcg
                best_model_weights = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ベストモデルに戻す
        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        # model.eval()
        # all_preds = []
        # with torch.no_grad():
        #     for X, y, cat_X, context_X, context_cat_X in val_loader:
        #         X, y, cat_X, context_X, context_cat_X = (
        #             X[0].to(device),
        #             y[0].to(device),
        #             cat_X[0].to(device),
        #             context_X[0].to(device),
        #             context_cat_X[0].to(device)
        #         )
        #         preds = model(X, cat_X, context_X, context_cat_X)  # shape: (馬数,) または (馬数, 1)
        #         preds = preds.squeeze()  # 余分な次元がある場合に対応
        #         prob = preds.cpu().numpy()
        #         # prob = torch.softmax(preds, dim=0).cpu().numpy()
        #         all_preds.append(prob)

        # valでの出力スコアを集める
        val_preds = []
        model.eval()
        with torch.no_grad():
            for X, y, cat_X, context_X, context_cat_X, win_labels, gain in val_loader:
                X, y, cat_X, context_X, context_cat_X, win_labels, gain = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device)
                preds = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                val_preds.append(preds)


        # ------------------------
        # Test評価
        # ------------------------
        test_scores = []
        with torch.no_grad():
            for X, y, cat_X, context_X, context_cat_X, win_labels, gain in test_loader:
                X, y, cat_X, context_X, context_cat_X, win_labels, gain = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device), win_labels[0].to(device), gain[0].to(device)
                raw_pred = model(X, cat_X, context_X, context_cat_X).squeeze().cpu().numpy()
                test_scores.append(raw_pred)

        # スコア付与
        val_df = val_df.copy()
        test_df = test_df.copy()
        test_df['pred_score'] = np.concatenate(test_scores)
        val_df['pred_score'] = np.concatenate(val_preds)

        test_df['expected_value'] = test_df['pred_score'] * test_df['オッズ']
        selected = test_df.loc[test_df.groupby('レースID')['expected_value'].idxmax()]

        total_bet = len(selected) * 100
        total_return = selected['単勝オッズ'].sum()

        hit_count = (selected['着順'] == 1).sum()
        roi = total_return / total_bet

        ndcg = calc_mean_ndcg(test_df)
        print(f"embedding_dim={emb_dim}, NDCG={ndcg:.4f}")

        print(f"\n[評価結果]")
        print(f"レース数: {len(selected)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(selected):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
        # print(test_df[['pred_score', 'オッズ', 'expected_value']].sort_values('expected_value', ascending=False).head(20))
        # print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])

        top = test_df.loc[test_df.groupby('レースID')['pred_score'].idxmax()]
        # top = top[top['pred_score'] * top['オッズ'] > 1.0]

        total_bet = len(top) * 100
        total_return = top['単勝オッズ'].sum()

        hit_count = (top['着順'] == 1).sum()
        roi = total_return / total_bet

        print(f"\n[top評価結果]")
        print(f"レース数: {len(top)}")
        print(f"的中数: {int(hit_count)}")
        print(f"的中率: {hit_count / len(top):.2%}")
        print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")

        val_df.to_csv(f'./csv/{field}_result_ranknet_val_{fold}.csv', index=False)
        test_df.to_csv(f'./csv/{field}_result_ranknet_test_{fold}.csv', index=False)

        # モデルを保存
        torch.save(model.state_dict(), f'./model/{field}_ranknet_{fold}.pth')


    # ['着順' '馬番' '斤量' '騎手' '人気' '単勝オッズ' '距離' 'フィールド' '馬場' '出走頭数' '馬単' 'レースID'
    #  '父馬' '間隔' '性' '齢' '1場所' '1過去着順' '1フィールド' '1距離' '1タイム' '1馬場' '1出走馬数' '1馬番'
    #  '1人気' '1斤量' '1コーナー通過順' '1後3F' '1馬体重' '1体重増減' '1着差' '1クラス' '1スピード指数'
    #  '1距離差' '1場所変化' '1フィールド変化' '2場所' '2過去着順' '2フィールド' '2距離' '2タイム' '2馬場'
    #  '2出走馬数' '2馬番' '2人気' '2斤量' '2コーナー通過順' '2後3F' '2馬体重' '2体重増減' '2着差' '2クラス'
    #  '2スピード指数' '2距離差' '2場所変化' '2フィールド変化' '3場所' '3過去着順' '3フィールド' '3距離' '3タイム'
    #  '3馬場' '3出走馬数' '3馬番' '3人気' '3斤量' '3コーナー通過順' '3後3F' '3馬体重' '3体重増減' '3着差'
    #  '3クラス' '3スピード指数' '3距離差' '3場所変化' '3フィールド変化' '4場所' '4過去着順' '4フィールド' '4距離'
    #  '4タイム' '4馬場' '4出走馬数' '4馬番' '4人気' '4斤量' '4コーナー通過順' '4後3F' '4馬体重' '4体重増減'
    #  '4着差' '4クラス' '4スピード指数' '4距離差' '4場所変化' '4フィールド変化' '5場所' '5過去着順' '5フィールド'
    #  '5距離' '5タイム' '5馬場' '5出走馬数' '5馬番' '5人気' '5斤量' '5コーナー通過順' '5後3F' '5馬体重'
    #  '5体重増減' '5着差' '5クラス' '5スピード指数' '5距離差' '5場所変化' '5フィールド変化' '平均クラス' '平均ペース'
    #  '1クラス差' '1ペース差' 'best着差' 'bestスピード指数' 'av着差' 'avスピード指数' '上昇度' 'オッズ'
    #  'rank']

    # feature_category = '距離', 'フィールド', '馬場', '出走頭数', '馬番', '性',
    # '1場所', '2場所', '3場所', '4場所', '5場所', '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド',
    # '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド', '1距離', '2距離', '3距離', '4距離', '5距離',
    # '1馬場', '2馬場', '3馬場', '4馬場', '5馬場','1コーナー通過順', '2コーナー通過順', '3コーナー通過順', '4コーナー通過順', '5コーナー通過順',
    # '1斤量', '2斤量', '3斤量', '4斤量', '5斤量', '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数', '1馬番', '2馬番', '3馬番', '4馬番', '5馬番',

