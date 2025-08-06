import copy
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import evaluation
from sklearn.preprocessing import StandardScaler

# 行・列ともに省略せず全て表示する設定
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# === 0. ハイパーパラメータ ===
n_splits = 5
num_epochs = 1000
batch_size = 1  # 1レースずつ処理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
place = '2'

common_cols = ['距離', 'フィールド', '馬場', '出走頭数', '平均クラス', '平均ペース']

context_cat_cols = ['フィールド', '馬場', '出走頭数']
context_num_cols = ['距離', '平均クラス', '平均ペース']

scale_cols = ['1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
            '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
            '斤量', '1斤量', '2斤量', '3斤量', '4斤量', '5斤量', '父馬', '間隔', '1馬体重', '2馬体重', '3馬体重', '4馬体重', '5馬体重',
            '1体重増減', '2体重増減', '3体重増減', '4体重増減', '5体重増減', '1スピード指数', '2スピード指数', '3スピード指数', '4スピード指数', '5スピード指数',
            'best着差', 'bestスピード指数', 'av着差', 'avスピード指数', '上昇度', '1クラス差', '1ペース差']

inversion_cols = ['人気', '齢', '間隔', '1着差', '2着差', '3着差', '4着差', '5着差', '1タイム', '2タイム', '3タイム', '4タイム', '5タイム',
                '1人気', '2人気', '3人気', '4人気', '5人気', 
                '1後3F', '2後3F', '3後3F', '4後3F', '5後3F', '1着差', '2着差', '3着差', '4着差', '5着差',
                'best着差', 'av着差']

feature_category = ['距離', 'フィールド', '馬場', '出走頭数', '馬番', '性',
                    '1場所', '2場所', '3場所', '4場所', '5場所', '1フィールド', '2フィールド', '3フィールド', '4フィールド', '5フィールド',
                    '1距離', '2距離', '3距離', '4距離', '5距離',
                    '1馬場', '2馬場', '3馬場', '4馬場', '5馬場','1コーナー通過順', '2コーナー通過順', '3コーナー通過順', '4コーナー通過順', '5コーナー通過順',
                    '1出走馬数', '2出走馬数', '3出走馬数', '4出走馬数', '5出走馬数', '1馬番', '2馬番', '3馬番', '4馬番', '5馬番']
diff_category_place = ['1場所変化', '2場所変化', '3場所変化', '4場所変化', '5場所変化']

diff_category_field = ['1フィールド変化', '2フィールド変化', '3フィールド変化', '4フィールド変化', '5フィールド変化']

embedding_cols = feature_category + diff_category_place + diff_category_field

# file_path
csv_path = './csv/df_all.csv'
df = evaluation.load_csv(csv_path)
# print(df.columns.values)
# print(df.head(10))

# Nanの処理
def fill_nan(df):
    # 1. NaN を -9999 で埋める
    df = df.fillna(-9999)

    # 2. 休養フラグの追加
    for i in range(1, 6):
        col_name = f'{i}過去着順'
        rest_flag = f'{i}休養'
        df[rest_flag] = (df[col_name] == -9999).astype(int)
    
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
        df[col] = df[col].astype(str) + '->' + place
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
eval_rank(df)

# === 1. データの前提 ===
# 1着の馬だけ 1、それ以外 0 の one-hot ターゲットを作成
# df['win_prob'] = (df['着順'] == 1).astype(float)

# rankから経験的勝率を計算する
df['rank'] = df.groupby('レースID')['score'].rank(method='first', ascending=False)

# 2. 実着順が1着（勝利）かどうかのフラグを作成（仮に '着順' カラムがあると仮定）
df['is_win'] = (df['着順'] == 1).astype(int)

# 3. 予測順位ごとに勝率を集計
rank_winrate = df.groupby('rank')['is_win'].mean().rename('win_prob_by_rank')
print(rank_winrate)

# 4. 各行に予測順位に応じた勝率をマージ
df = df.merge(rank_winrate, how='left', left_on='rank', right_index=True)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 数値安定化
    return e_x / e_x.sum()

# 5. 最終的な win_prob を追加
df['win_prob'] = df['win_prob_by_rank']
# df['win_prob'] = df.groupby('レースID')['win_prob'].transform(softmax)

df['win_prob'] = df.groupby('レースID')['win_prob'].transform(lambda x: x / x.sum())  # 正規化
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

# レースごとに softmax を適用するため、例として 'レースID' 単位で groupby
# df['win_prob'] = df.groupby('レースID')['着順'].transform(softmax_neg_rank)

# Nanの処理
df = fill_nan(df)
df = race_feature(df)
df = inversion(df)

feature_cols = [col for col in df.columns if col not in ['レースID', '着順', 'rank', 'オッズ', '単勝オッズ', '馬単', 'score', 'win_flag', 'win_prob', 'is_win', 'win_prob_by_rank']]
group_col = 'レースID'
target_col = 'win_prob'

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# print(df.head(10))

feature_cols = [col for col in feature_cols if col not in embedding_cols and col not in common_cols]

# === 2. Dataset定義 ===
class RaceDataset(Dataset):
    def __init__(self, X_groups, y_groups, cat_groups, context_num_groups, context_cat_groups):
        self.X_groups = X_groups
        self.y_groups = y_groups
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
    )

# collate_fn の追加
def collate_fn(batch):
    # 最大頭数（最大系列長）を取得
    max_len = max(item["length"] for item in batch)

    def pad_tensor(tensor, pad_len, pad_value=0):
        pad_size = (0, 0, 0, pad_len - tensor.size(0))  # (dim2_pad, dim2_pad, dim1_pad, dim1_pad)
        return nn.functional.pad(tensor, pad_size, value=pad_value)

    x = torch.stack([pad_tensor(item["x"], max_len) for item in batch])
    cat_x = torch.stack([pad_tensor(item["cat_x"], max_len) for item in batch])
    y = torch.stack([pad_tensor(item["y"].unsqueeze(-1), max_len).squeeze(-1) for item in batch])
    context_num = torch.stack([pad_tensor(item["context_num"], max_len) for item in batch])
    context_cat = torch.stack([pad_tensor(item["context_cat"], max_len) for item in batch])

    # Attention mask: 1 for actual data, 0 for padded
    attention_mask = torch.tensor([
        [1] * item["length"] + [0] * (max_len - item["length"]) for item in batch
    ])

    return x, y, cat_x, context_num, context_cat, attention_mask

class SelfAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)  # 残差接続

# === 3. モデル定義 ===
class ListNet(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64, attn_dim=128, num_heads=4):
        super().__init__()
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

        # === Attention 追加 ===
        # Attention前の次元（数値 + embedding + context）
        self.raw_input_dim = num_features + len(embedding_sizes) * emb_dim + hidden_dim
        self.input_projection = nn.Linear(self.raw_input_dim, attn_dim)  # Attention用に次元調整

        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)

        self.context_projection = nn.Linear(hidden_dim, attn_dim)

        # 最終予測 MLP
        self.fc = nn.Sequential(
            nn.Linear(attn_dim, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, cat_X, context_num, context_cat):
        # 埋め込み層処理
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)

        # context embedding（1サンプルのみ使えばOK）
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)  # [emb_dim * num_context_cat_features]

        # context 数値特徴（1サンプル分）
        context_num = context_num[0]  # [num_context_num_features]

        # 結合 → MLP
        context_all = torch.cat([context_num, context_emb], dim=0)  # [context_input_dim]
        context_out = self.context_fc(context_all.unsqueeze(0))  # [1, hidden_dim]

        # 各馬に同じcontextを与える
        context_expand = context_out.expand(x.size(0), -1)  # [頭数, hidden_dim]

        # # ========== 入力結合（数値 + emb + context） ==========
        full_input = torch.cat([x, emb, context_expand], dim=1)  # [頭数, raw_input_dim]
        x_proj = self.input_projection(full_input).unsqueeze(0)  # [1, 頭数, attn_dim]

        # # ========== Self-Attention ==========
        # attn_out, _ = self.attn(x_proj, x_proj, x_proj)          # [1, 頭数, attn_dim]
        # attn_out = attn_out.squeeze(0)                           # [頭数, attn_dim]

        # x_proj: [1, 頭数, attn_dim]
        # context_out: [1, hidden_dim]
        
        context_proj = self.context_projection(context_out)  # [1, attn_dim]
        context_proj = context_proj.unsqueeze(0)  # [1, 1, attn_dim]


        x_with_context = torch.cat([x_proj, context_proj], dim=1)  # [1, 頭数+1, attn_dim]

        attn_out, _ = self.attn(x_with_context, x_with_context, x_with_context)  # [1, 頭数+1, attn_dim]
        attn_out = attn_out[:, :-1, :]  # context を除いた部分を出力とする
        attn_out = attn_out.squeeze(0)  # [頭数, attn_dim] に変換
        out = self.fc(attn_out).squeeze(-1)  # [頭数]
        return out
        # ========== 最終出力 ==========
        return self.fc(attn_out).squeeze(-1)                     # [頭数]

# === 4. Listwise loss ===
def listnet_loss(preds, labels):
    # print(f'preds: {preds}')
    # pred_rank = preds.argsort(descending=True)
    # true_rank = labels.argsort(descending=True)

    # print("予測1位の馬番:", pred_rank[0].item())
    # print("実際1位の馬番:", true_rank[0].item())
    preds = preds - preds.max()
    # labels = labels - labels.max()
    # P_y = torch.softmax(labels, dim=0)
    # P_y = labels
    P_z = torch.softmax(preds, dim=0)
    # loss = -torch.sum(P_y * torch.log(P_z + 1e-12))  # ここを変更
    loss = -torch.sum(labels * torch.log(P_z + 1e-12))
    # print(f'P_z: {P_z}')
    return loss

def evaluate_model_on_val_df(val_df, model_path, fold=0):
    val_df = val_df.copy()
    def make_softmax_with_temperature(T=1.0):
        def softmax(x):
            x = x / T
            e_x = np.exp(x - np.max(x))  # 安定化のために最大値を引く
            return e_x / e_x.sum()
        return softmax
    # df_sorted = val_df.sort_values(by=['レースID', 'pred_score'], ascending=[True, False])
    # print(df_sorted[['レースID', 'pred_score']].head(30))
    
    # val_df['log_odds'] = np.log(val_df['オッズ'] + 1)
    # for i in range(1, 6):
    #     T = 0.5 * i  # 例：温度を0.5に設定（小さいほど尖る）
    #     softmax_T = make_softmax_with_temperature(T)

    #     val_df['softmax_score'] = val_df.groupby('レースID')['pred_score'].transform(softmax_T)
    #     val_df['expected_value'] = val_df['softmax_score'] * val_df['log_odds']
    #     top_by_race = val_df.groupby('レースID').apply(
    #         lambda df: df.sort_values('expected_value', ascending=False)
    #     )

    #     print(top_by_race[['softmax_score', 'オッズ', 'expected_value', 'win_prob']].head(100))

    a = 0.5
    b = 0.5
    val_df['expected_value'] = (val_df['pred_score'] ** a) * (val_df['オッズ'] ** b)
        
    selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]
    total_bet = len(selected) * 100
    total_return = selected['単勝オッズ'].sum()
    hit_count = (selected['着順'] == 1).sum()
    roi = total_return / total_bet

    print(f"\n[評価結果 - Fold {fold}]")
    print(f"レース数: {len(selected)}")
    print(f"的中数: {int(hit_count)}")
    print(f"的中率: {hit_count / len(selected):.2%}")
    print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
    # print(selected[['softmax_score', 'log_odds', 'expected_value']].sort_values('expected_value', ascending=False).head(20))

    
    # top = top[top['expected_value'] > 100]

    # for i in range(1, 6):
    #     top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
    #     top = top[top['expected_value'] > i]
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


# === 5. KFold処理 ===
gkf = GroupKFold(n_splits=n_splits)
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df[target_col], groups=df[group_col])):
    print(f"\nFold {fold+1}")

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    def group_by_race(df_part):
        X_groups, y_groups, cat_groups, context_num_groups, context_cat_groups = [], [], [], [], []
        for _, g in df_part.groupby(group_col):
            X = g[feature_cols].values.astype(np.float32)
            y = g[target_col].values.astype(np.float32)
            cat_X = g[embedding_cols].values.astype(np.int64)

            # context内の数値・カテゴリに分ける（レースごとに1行だけ使う）
            context_num = g[context_num_cols].iloc[0].values.astype(np.float32)  # shape: [context_num_dim]
            context_cat = g[context_cat_cols].iloc[0].values.astype(np.int64)    # shape: [context_cat_dim]

            num_horses = len(g)
            context_num = np.tile(context_num, (num_horses, 1))  # shape: [頭数, context_num_dim]
            context_cat = np.tile(context_cat, (num_horses, 1))  # shape: [頭数, context_cat_dim]

            X_groups.append(torch.tensor(X, dtype=torch.float32))
            y_groups.append(torch.tensor(y, dtype=torch.float32))
            cat_groups.append(torch.tensor(cat_X, dtype=torch.long))
            context_num_groups.append(torch.tensor(context_num, dtype=torch.float32))
            context_cat_groups.append(torch.tensor(context_cat, dtype=torch.long))

        return X_groups, y_groups, cat_groups, context_num_groups, context_cat_groups

    X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups = group_by_race(train_df)
    X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups = group_by_race(val_df)

    train_dataset = RaceDataset(X_train_groups, y_train_groups, cat_train_groups, context_train_num_groups, context_train_cat_groups)
    val_dataset = RaceDataset(X_val_groups, y_val_groups, cat_val_groups, context_val_num_groups, context_val_cat_groups)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    embedding_sizes = [df[col].nunique() + 1 for col in embedding_cols]  # 各カテゴリ列のクラス数
    context_embedding_sizes = [df[col].nunique() + 1 for col in context_cat_cols]  # 各カテゴリ列のクラス数

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


    # model = ListNet(embedding_sizes=embedding_sizes, num_features=len(feature_cols), emb_dim=4)
    # model.apply(init_weights)
    # model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # for emb_dim in [16, 24, 32]:
    emb_dim = 16
    model = ListNet(embedding_sizes=embedding_sizes, num_features=len(feature_cols), context_embedding_sizes=context_embedding_sizes, context_num_sizes=len(context_num_cols), emb_dim=emb_dim)
    model.apply(init_weights)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    # ハイパーパラメータ
    patience = 7  # 何エポック改善がなければ終了するか
    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_weights = None
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X, y, cat_X, context_X, context_cat_X in train_loader:
            X, y, cat_X, context_X, context_cat_X = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device)
            y_sum = y.detach().cpu().numpy().sum()
            # print(f"Label sum (should be 1.0): {y_sum:.5f}")  # 1.0に近いか確認
            preds = model(X, cat_X, context_X, context_cat_X)
            loss = listnet_loss(preds, y)
            # デバッグ出力
            # print(f"preds: {preds.detach().cpu().numpy()}")
            # print(f"labels: {y.detach().cpu().numpy()}")
            # print(f"loss: {listnet_loss(preds, y).item()}")
            # print(f"preds mean: {preds.mean().item():.3f}, std: {preds.std().item():.3f}")

            # デバッグ出力 可視化
            pred_probs = torch.softmax(preds, dim=0).detach().cpu().numpy()
            labels = y.cpu().numpy()

            # plt.figure(figsize=(8,4))
            # plt.plot(pred_probs, label='predicted prob')
            # plt.plot(labels, label='label prob')
            # plt.legend()
            # plt.title("Prediction vs Label Distribution")
            # plt.show()

            optimizer.zero_grad()
            loss.backward()

            # 勾配確認コード（例）
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
            #     else:
            #         print(f"{name}: grad is None")


            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Train Loss: {total_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y, cat_X, context_X, context_cat_X in val_loader:
                X, y, cat_X, context_X, context_cat_X = X[0].to(device), y[0].to(device), cat_X[0].to(device), context_X[0].to(device), context_cat_X[0].to(device)
                preds = model(X, cat_X, context_X, context_cat_X)
                loss = listnet_loss(preds, y)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early Stopping 判定
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
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

    model.eval()
    all_preds = []
    with torch.no_grad():
        for X, y, cat_X, context_X, context_cat_X in val_loader:
            X, y, cat_X, context_X, context_cat_X = (
                X[0].to(device),
                y[0].to(device),
                cat_X[0].to(device),
                context_X[0].to(device),
                context_cat_X[0].to(device)
            )
            preds = model(X, cat_X, context_X, context_cat_X)  # shape: (馬数,) または (馬数, 1)
            preds = preds.squeeze()  # 余分な次元がある場合に対応
            prob = torch.softmax(preds, dim=0).cpu().numpy()
            all_preds.append(prob)

    # スコア付与
    val_df = val_df.copy()
    val_df['pred_score'] = np.concatenate(all_preds)

    val_df['expected_value'] = val_df['pred_score'] * val_df['オッズ']
    selected = val_df.loc[val_df.groupby('レースID')['expected_value'].idxmax()]

    total_bet = len(selected) * 100
    total_return = selected['単勝オッズ'].sum()

    hit_count = (selected['着順'] == 1).sum()
    roi = total_return / total_bet

    ndcg = calc_mean_ndcg(val_df)
    print(f"embedding_dim={emb_dim}, NDCG={ndcg:.4f}")

    print(f"\n[評価結果]")
    print(f"レース数: {len(selected)}")
    print(f"的中数: {int(hit_count)}")
    print(f"的中率: {hit_count / len(selected):.2%}")
    print(f"回収率: {roi:.2%}（{total_return:.0f}円 / {total_bet}円）")
    print(val_df[['pred_score', 'オッズ', 'expected_value']].sort_values('expected_value', ascending=False).head(20))
    print(selected[selected['着順'] == 1][['pred_score', 'オッズ', 'expected_value']])

    top = val_df.loc[val_df.groupby('レースID')['pred_score'].idxmax()]
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

    val_df.to_csv(f'./csv/tokyo_result_listnet_{fold}.csv', index=False)

    # モデルを保存
    torch.save(model.state_dict(), f'./model/tokyo_listnet_{fold}.pth')


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

