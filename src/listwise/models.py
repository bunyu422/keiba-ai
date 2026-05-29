import torch
import torch.nn as nn

# ListNet2: コンテキスト（レース情報）と馬特徴を融合するベースモデル（ゲート付き）
class ListNet2(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64):
        super().__init__()
        # 馬個別カテゴリ特徴 → Embedding
        self.embedding_sizes = embedding_sizes
        self.context_embedding_sizes = context_embedding_sizes
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in embedding_sizes
        ])
        # コンテキストカテゴリ特徴 → Embedding
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in context_embedding_sizes
        ])
        # コンテキスト全特徴（数値+カテゴリ埋め込み）→ 隠れ層
        context_input_dim = context_num_sizes + len(context_embedding_sizes) * emb_dim
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 馬特徴をコンテキスト次元に合わせる射影
        horse_input_dim = num_features + len(embedding_sizes) * emb_dim
        self.horse_proj = nn.Linear(horse_input_dim, hidden_dim)

        # スコアリングネットワーク（馬特徴×コンテキストの相互作用込み）
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 1)
        )
        # 順位ゲート（rank_scores を元に馬特徴の加算を制御）
        self.rank_gate = nn.Linear(1, hidden_dim)
        # 残差結合
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, 1)
        )

    def forward(self, x, cat_X, context_num, context_cat, rank_scores=None):
        # 馬カテゴリ特徴の埋め込み＋数値特徴と結合
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        horse_features = torch.cat([x, emb], dim=1)

        # コンテキスト特徴の埋め込み＋全結合
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)
        context_num = context_num[0]
        context_all = torch.cat([context_num, context_emb], dim=0)
        context_out = self.context_fc(context_all.unsqueeze(0))
        context_expand = context_out.expand(horse_features.size(0), -1)  # 頭数分ブロードキャスト

        # 次元不一致時は馬特徴を射影
        if horse_features.size(1) != context_expand.size(1):
            horse_features = self.horse_proj(horse_features)

        # 順位スコアをゲートに使った加算（←過去順位が高い馬ほどコンテキストを強く反映）
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)
            norm_rank = norm_rank.unsqueeze(1)
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = horse_features + gate * context_expand
        else:
            horse_features = horse_features + context_expand

        # 馬特徴・コンテキスト・相互作用を結合してスコアリング
        interaction = horse_features * context_expand
        combined = torch.cat([horse_features, context_expand, interaction], dim=1)
        out = self.fc(combined)
        out = out + self.residual_proj(combined)
        return out.squeeze(-1)

# ListNet（拡張版）: 5種類の相互作用（積・和・絶対差）を使う
class ListNet(nn.Module):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes, emb_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding_sizes = embedding_sizes
        self.context_embedding_sizes = context_embedding_sizes
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in embedding_sizes
        ])
        self.context_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, emb_dim) for num_classes in context_embedding_sizes
        ])
        context_input_dim = context_num_sizes + len(context_embedding_sizes) * emb_dim
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        horse_input_dim = num_features + len(embedding_sizes) * emb_dim
        self.horse_proj = nn.Linear(horse_input_dim, hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 5, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 1)
        )
        self.rank_gate = nn.Linear(1, hidden_dim)
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*3),
            nn.Linear(hidden_dim*3, 1)
        )
        # 馬特徴専用全結合＋残差
        self.horse_fc = nn.Sequential(
            nn.Linear(horse_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.horse_residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, cat_X, context_num, context_cat, rank_scores=None):
        # 馬特徴 → HorseFC → 隠れベクトル
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        horse_features = torch.cat([x, emb], dim=1)
        horse_features = self.horse_fc(horse_features)

        # コンテキスト特徴
        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)
        context_num = context_num[0]
        context_all = torch.cat([context_num, context_emb], dim=0)
        context_out = self.context_fc(context_all.unsqueeze(0))
        context_expand = context_out.expand(horse_features.size(0), -1)

        # 順位ゲート付き加算
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)
            norm_rank = norm_rank.unsqueeze(1)
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = horse_features + gate * context_expand
        else:
            horse_features = horse_features + context_expand

        # 5種類の相互作用: 積, 和, 絶対差
        mul_interaction = horse_features * context_expand
        add_interaction = horse_features + context_expand
        diff_interaction = torch.abs(horse_features - context_expand)
        combined = torch.cat([
            horse_features,
            context_expand,
            mul_interaction,
            add_interaction,
            diff_interaction
        ], dim=1)
        out = self.fc(combined)
        resid = self.horse_residual(horse_features)
        out = out + resid
        return out.squeeze(-1)

# ListNetGRU: ListNet に GRU（過去出走履歴）を追加
class ListNetGRU(ListNet):
    def __init__(self, embedding_sizes, num_features, context_embedding_sizes, context_num_sizes,
                 emb_dim=16, hidden_dim=64, gru_hidden=32, num_past_runs=5, past_feat_dim=5):
        super().__init__(embedding_sizes, num_features, context_embedding_sizes, context_num_sizes,
                         emb_dim, hidden_dim)
        self.num_past_runs = num_past_runs
        self.past_feat_dim = past_feat_dim
        # GRU: 過去出走の時系列パターンを学習
        self.gru = nn.GRU(input_size=past_feat_dim, hidden_size=gru_hidden, batch_first=True)
        self.horse_proj = nn.Linear(num_features + len(embedding_sizes) * emb_dim + gru_hidden, hidden_dim)

    def forward(self, x, cat_X, context_num, context_cat, past_runs=None, rank_scores=None):
        emb = [emb_layer(cat_X[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=1)
        horse_features = torch.cat([x, emb], dim=1)

        # 過去出走データがあればGRUに通し、最終隠れ状態を結合
        if past_runs is not None:
            gru_out, _ = self.gru(past_runs)
            gru_last = gru_out[:, -1, :]
            horse_features = torch.cat([horse_features, gru_last], dim=1)

        context_emb = [emb_layer(context_cat[0, i]) for i, emb_layer in enumerate(self.context_embeddings)]
        context_emb = torch.cat(context_emb, dim=0)
        context_num = context_num[0]
        context_all = torch.cat([context_num, context_emb], dim=0)
        context_out = self.context_fc(context_all.unsqueeze(0))
        context_expand = context_out.expand(horse_features.size(0), -1)

        # 順位ゲート＋馬特徴射影
        if rank_scores is not None:
            min_r = rank_scores.min()
            max_r = rank_scores.max()
            norm_rank = 1 - (rank_scores - min_r) / (max_r - min_r + 1e-12)
            norm_rank = norm_rank.unsqueeze(1)
            gate = torch.sigmoid(self.rank_gate(norm_rank))
            horse_features = self.horse_proj(horse_features)
            horse_features = horse_features + gate * context_expand
        else:
            horse_features = self.horse_proj(horse_features)
            horse_features = horse_features + context_expand

        interaction = horse_features * context_expand
        combined = torch.cat([horse_features, context_expand, interaction], dim=1)
        out = self.fc(combined)
        out = out + self.residual_proj(combined)
        out = torch.tanh(out) * 5  # 出力を[-5, 5]に制限
        return out.squeeze(-1)
