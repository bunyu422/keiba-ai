from torch.utils.data import Dataset, DataLoader

# 1レース＝1サンプルのデータセット
class RaceDataset(Dataset):
    def __init__(self, X_groups, y_groups,
                 cat_groups, context_num_groups, context_cat_groups, win_groups, payout_groups, win_index_groups, past_groups=None):
        # X_groups: [レース数, 頭数, 特徴量数] の数値特徴
        # y_groups: [レース数, 頭数] の着順ラベル
        # cat_groups: [レース数, 頭数, カテゴリ数] のembedding用カテゴリID
        # context_num_groups: [レース数, 頭数, コンテキスト数値数] レース共通数値特徴
        # context_cat_groups: [レース数, 頭数, コンテキストカテゴリ数] レース共通カテゴリ特徴
        # win_groups: [レース数, 頭数] 0/1 勝ちフラグ
        # payout_groups: [レース数, 頭数] 配当
        # win_index_groups: [レース数] 勝ち馬のインデックス
        self.X_groups = X_groups
        self.y_groups = y_groups
        self.win_groups = win_groups
        self.payout_groups = payout_groups
        self.cat_groups = cat_groups
        self.context_num_groups = context_num_groups
        self.context_cat_groups = context_cat_groups
        self.win_index_groups = win_index_groups

    def __len__(self):
        return len(self.X_groups)

    def __getitem__(self, idx):
        return (
            self.X_groups[idx],
            self.y_groups[idx],
            self.cat_groups[idx],
            self.context_num_groups[idx],
            self.context_cat_groups[idx],
            self.win_groups[idx],
            self.payout_groups[idx],
            self.win_index_groups[idx],
        )
