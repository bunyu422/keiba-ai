import numpy as np
import pandas as pd
from collections import defaultdict

# -----------------------
# 1) 簡易 LinUCB 実装
# -----------------------
class LinUCB:
    def __init__(self, n_arms, dim, alpha=1.0):
        """
        n_arms: number of policies
        dim: dimension of context vector
        alpha: exploration parameter (higher => more exploration)
        """
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        # For each arm: A = D x D identity, b = D vector
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

    def select_arm(self, x):
        # x: context vector (dim,)
        p_vals = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv.dot(self.b[a])
            exploit = theta.dot(x)
            explore = self.alpha * np.sqrt(x.dot(A_inv).dot(x))
            p = exploit + explore
            p_vals.append(p)
        return int(np.argmax(p_vals)), p_vals

    def update(self, arm, x, reward):
        # reward: scalar
        x = x.reshape(-1)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

# -----------------------
# 2) ポリシー定義（候補）
# -----------------------
# policies take 'race_df' (rows for a single race) and return chosen horse_id and meta info
def policy_ev_max(race_df, p_col="softmax_score", odds_col="オッズ", race_col="レースID"):
    race_df = race_df.copy()
    race_df['EV'] = race_df[p_col] * race_df[odds_col]
    pick = race_df.sort_values('EV', ascending=False).iloc[[0]]
    return pick

def policy_prob_max(race_df, p_col="pred_score"):
    pick = race_df.sort_values(p_col, ascending=False).iloc[[0]]
    return pick

def policy_odds_range(race_df, odds_col="オッズ", low=3.0, high=8.0, p_col="pred_score"):
    # まずオッズが指定レンジの馬を探す。なければ top prob を返す
    candidates = race_df[(race_df[odds_col] >= low) & (race_df[odds_col] <= high)]
    if len(candidates) == 0:
        return policy_prob_max(race_df, p_col=p_col)
    # その中でpred_prob最高を選ぶ
    return candidates.sort_values(p_col, ascending=False).iloc[[0]]

# Add other policies as needed...

# -----------------------
# 3) レース文脈抽出関数
# -----------------------
def extract_race_context(race_df):
    """
    race_df: DataFrame containing all horses of the race.
    Returns: numpy vector (context)
    Candidate features:
      - n_horses
      - distance (could be numeric or bucket embedding; here numeric normalized)
      - track_condition (encoded numeric)
      - odds_std (std of odds)
      - top_odds_ratio (odds of 1st / odds of 2nd)  (popularity concentration)
    You can expand/normalize as needed.
    """
    n_horses = len(race_df)
    # assume '距離' and '馬場' columns exist, otherwise adapt
    distance = race_df['距離'].iloc[0] if '距離' in race_df.columns else 0
    # horse-level odds -> summary
    odds = race_df['オッズ'].values if 'オッズ' in race_df.columns else np.ones(n_horses)
    odds_std = float(np.std(odds))
    sorted_odds = np.sort(odds)
    top_ratio = float(sorted_odds[0] / (sorted_odds[1] + 1e-9)) if n_horses > 1 else 1.0
    # track cond numeric: map common text to number (extend map as needed)
    if '馬場' in race_df.columns:
        track_cond = race_df['馬場'].iloc[0]
        # cond_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
        # track_cond = cond_map.get(cond, 0)
    else:
        track_cond = 0
    # Normalize/scale simple heuristics (you can fit scalers on training set)
    return np.array([n_horses, distance, track_cond, odds_std, top_ratio], dtype=float)

# -----------------------
# 4) 報酬関数（例）
# -----------------------
def calc_reward_for_choice(chosen_row, df_payout, stake=100, bet_type="単勝"):
    """
    chosen_row: pandas Series representing the chosen horse within race
    df_payout: DataFrame keyed by race and horse giving real payout info (you might already have this)
    stake: stake amount for this bet
    Returns: reward scalar (e.g., profit in yen normalized)
    """
    race_id = chosen_row['レースID']
    horse_no = chosen_row['馬番']
    # find payout row; your df_payout schema may differ
    # assume df_payout has index (race_id, horse_no) with 'payout' for 単勝, or compute from results
    try:
        payout_row = df_payout.loc[(race_id, horse_no)]
        # payout amount multiplier: ex. 単勝 payout 120 -> you receive 120 per 100 stake? adjust to your dataset's meaning
        payout_multiplier = payout_row['オッズ'] * payout_row['is_win']  # define schema accordingly
        profit = stake * (payout_multiplier - 1.0)
    except Exception:
        # if not found, assume loss
        profit = -stake
    return profit

def calc_reward_for_choice_A(chosen_row, df_payout, stake=100, lam=0.1, scale_ev=10.0):
    """
    Reward A:
      reward = hit(0/1) + λ * normalized_EV
    """
    race_id = chosen_row['レースID']
    horse_no = chosen_row['馬番']

    try:
        payout_row = df_payout.loc[(race_id, horse_no)]
        is_win = int(payout_row['is_win'] == 1)
        odds = payout_row['オッズ']
    except KeyError:
        is_win = 0
        odds = chosen_row.get("オッズ", 0.0)

    # EVを計算（pred_score は事前にsoftmax済み or 勝率予測）
    ev = chosen_row["softmax_score"] * odds
    normalized_ev = ev / scale_ev  # スケーリングして極端に大きくならないようにする

    reward = is_win + lam * normalized_ev
    return reward


def calc_reward_for_choice_B(chosen_row, df_payout, stake=100, scale=1000.0):
    """
    Reward B:
      reward = tanh(profit_yen / scale)
    """
    race_id = chosen_row['レースID']
    horse_no = chosen_row['馬番']

    try:
        payout_row = df_payout.loc[(race_id, horse_no)]
        if payout_row['is_win'] == 1:
            profit = stake * (payout_row['オッズ'] - 1.0)
        else:
            profit = -stake
    except KeyError:
        profit = -stake

    reward = np.tanh(profit / scale)
    return reward

# -----------------------
# 5) バンディットを使ったオフラインシミュレーション（1レースずつ順に回す）
# -----------------------
def run_contextual_bandit_simulation(
    df, df_payout,
    policies,  # list of (name, function, policy_kwargs)
    context_dim=5,
    alpha=1.0,
    stake_map=None,  # dict policy_name -> stake multiplier (relative)
    initial_A=None
):
    """
    Returns: summary stats per policy and overall results DataFrame (each bet record)
    """
    sel = pd.DataFrame()
    n_arms = len(policies)
    lb = LinUCB(n_arms=n_arms, dim=context_dim, alpha=alpha)

    records = []
    policy_name_to_idx = {policies[i][0]: i for i in range(len(policies))}
    # stats
    stats = defaultdict(lambda: {'bets': 0, 'profit': 0.0})

    races_ordered = df['レースID'].unique().tolist()

    for race_id in races_ordered:
        race_df = df[df['レースID'] == race_id].reset_index(drop=True)
        if race_df.empty:
            continue
        # --- get context ---
        ctx = extract_race_context(race_df)
        # normalize context optionally (recommended to scale by training set - here crude scaling)
        # select arm
        arm, p_vals = lb.select_arm(ctx)
        policy_name, policy_fn, policy_kwargs = policies[arm]
        chosen = policy_fn(race_df, **(policy_kwargs or {}))
        # compute stake for this policy
        stake = 100 * (stake_map.get(policy_name, 1.0) if stake_map else 1.0)
        # compute reward (profit in yen)
        reward_yen = calc_reward_for_choice(chosen, df_payout, stake=stake)
        # Option: normalize reward when updating bandit (e.g., divide by max stake or clip)
        # here we normalize to a roughly bounded value: reward_norm = reward_yen / 1000
        reward_norm = reward_yen / 1000.0
        lb.update(arm, ctx, reward_norm)

        # record
        # records.append({
        #     'race_id': race_id,
        #     'policy': policy_name,
        #     'horse_no': chosen['馬番'],
        #     'stake': stake,
        #     'profit_yen': reward_yen,
        #     'reward_norm': reward_norm,
        #     'p_vals': p_vals
        # })
        # stats[policy_name]['bets'] += 1
        # stats[policy_name]['profit'] += reward_yen
        sel = pd.concat([sel, chosen])

    # records_df = pd.DataFrame(records)
    return sel

# -----------------------
# 6) grid_search_ev_policy と統合する方法（概念）
# -----------------------
# あなたの grid_search_ev_policy 内で `filter_by_thresholds` 後に sel (選択分) がある部分に次を差し込むイメージ:
#
# 1) main_selected = sel (Primary)  <- 既存処理
# 2) dropped_races = set(base_races) - set(main_selected_races)
# 3) for each dropped race (in chronological order if offline):
#       ctx = extract_race_context(...)
#       arm = bandit.select_arm(ctx)
#       chosen = policies[arm](...)
#       stake = base_stake * stake_map[policy_name]
#       compute reward using df_payout and update bandit
#       append chosen to sel (with a column logic=policy_name)
#
# 4) 最後に evaluate_selection(sel, df_payout=..., stake=...) で全体評価
#
# これにより "Primary を確保しつつ、落ちたレースは Bandit が補完" する形になります。
#
# 実装のポイント：
# - オフラインの学習ループは「時系列順（過去→未来）」で回して bandit を更新する
# - evaluate_selection は主に総合評価に使い、bandit の更新はレース単位で reward を即時反映
# - stake_map を用いて Primary は stake=1.0、補完は 0.2 などにしてリスクコントロール

