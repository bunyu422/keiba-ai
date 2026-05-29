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

def policy_ev_prob_combo(race_df, alpha=0.5, p_col="pred_score", odds_col="オッズ"):
    race_df = race_df.copy()
    race_df['score'] = alpha * (race_df[p_col]*race_df[odds_col]) + (1-alpha) * race_df[p_col]
    return race_df.sort_values('score', ascending=False).iloc[[0]]

def policy_top3_softmax_pick(race_df, score_col="softmax_score"):
    race_df = race_df.copy()
    # 上位3頭だけ抽出（スコア順）
    top3 = race_df.sort_values(score_col, ascending=False).head(3)
    # softmax_scoreはすでに確率なので、そのまま確率として使用
    probs = top3[score_col].to_numpy()
    probs = probs / probs.sum()  # 念のため合計1に正規化
    choice_idx = np.random.choice(len(top3), p=probs)
    return top3.iloc[[choice_idx]]

def policy_low_odds(race_df, odds_col="オッズ", max_odds=4.0, p_col="pred_score"):
    candidates = race_df[race_df[odds_col] <= max_odds]
    if len(candidates) == 0:
        return policy_prob_max(race_df, p_col)
    return candidates.sort_values(p_col, ascending=False).iloc[[0]]

def policy_fastest_last3f(race_df, col='av後3F'):
    """
    最後3Fが最も小さい馬を選択
    """
    return race_df.sort_values(col, ascending=True).iloc[[0]]

def policy_highest_speed_index(race_df, col='avスピード指数'):
    """
    スピード指数が最も大きい馬を選択
    """
    return race_df.sort_values(col, ascending=False).iloc[[0]]

def policy_highest_uptrend(race_df, col='上昇度'):
    """
    上昇度が最も大きい馬を選択
    """
    return race_df.sort_values(col, ascending=False).iloc[[0]]

def policy_main_filter(
    race_df,
    ev_min=1.1,
    prob_min=0.0,
    odds_min=3.0,
    odds_max=float('inf'),
    ev_max=3,
    p_col="softmax_score",
    odds_col="オッズ"
):
    """
    mainフィルタ相当の条件をポリシーとして定義
    条件を満たす馬から期待値最大を選ぶ
    条件を満たす馬がいなければ「勝率最大」にフォールバック
    """
    race_df = race_df.copy()
    race_df["expected_value"] = race_df[p_col] * race_df[odds_col]

    sel = race_df[
        (race_df["expected_value"] >= ev_min) &
        (race_df["expected_value"] <= ev_max) &
        (race_df[p_col] >= prob_min) &
        (race_df[odds_col] >= odds_min) &
        (race_df[odds_col] <= odds_max)
    ]

    if len(sel) > 0:
        return sel.sort_values("expected_value", ascending=False).iloc[[0]]
    else:
        # fallback → 勝率最大馬
        return race_df.sort_values(p_col, ascending=False).iloc[[0]]

# Add other policies as needed...

# -----------------------
# 3) レース文脈抽出関数
# -----------------------
# def extract_race_context(race_df):
#     """
#     race_df: DataFrame containing all horses of the race.
#     Returns: numpy vector (context)
#     Candidate features:
#       - n_horses
#       - distance (could be numeric or bucket embedding; here numeric normalized)
#       - track_condition (encoded numeric)
#       - odds_std (std of odds)
#       - top_odds_ratio (odds of 1st / odds of 2nd)  (popularity concentration)
#     You can expand/normalize as needed.
#     """
#     n_horses = len(race_df)
#     # assume '距離' and '馬場' columns exist, otherwise adapt
#     distance = race_df['距離'].iloc[0] if '距離' in race_df.columns else 0
#     # horse-level odds -> summary
#     odds = race_df['オッズ'].values if 'オッズ' in race_df.columns else np.ones(n_horses)
#     odds_std = float(np.std(odds))
#     sorted_odds = np.sort(odds)
#     top_ratio = float(sorted_odds[0] / (sorted_odds[1] + 1e-9)) if n_horses > 1 else 1.0
#     # track cond numeric: map common text to number (extend map as needed)
#     if '馬場' in race_df.columns:
#         track_cond = race_df['馬場'].iloc[0]
#         # cond_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
#         # track_cond = cond_map.get(cond, 0)
#     else:
#         track_cond = 0
#     # Normalize/scale simple heuristics (you can fit scalers on training set)
#     return np.array([n_horses, distance, track_cond, odds_std, top_ratio], dtype=float)

def extract_race_context(race_df):
    """
    レースDFからバンディット用のコンテキストベクトルを作成
    含まれる情報：
    - n_horses（出走頭数）
    - オッズ分布: odds_std, fav_odds, fav_prob, longshot_ratio
    - 各戦略トップ馬: prob, ev, odds
    - トップ馬の飛び抜け度: 2番手との差 diff2, Zスコア zscore
      対象: softmax_score, expected_value, 後3F, スピード指数, 上昇度
    """
    n_horses = len(race_df)

    # --- オッズ分布情報 ---
    odds = race_df['オッズ'].values
    odds_std = float(np.std(odds))
    fav_odds = float(np.min(odds))
    fav_prob = float(race_df['softmax_score'].max())
    longshot_ratio = float(np.mean(odds > 20))

    # --- トップ馬 summary 関数 ---
    def get_top_features(df, col, maximize=True):
        """col の最大/最小馬を取り出し、飛び抜け度を計算"""
        if col not in df.columns:
            return {"prob":0,"ev":0,"odds":0,"value":0,"diff2":0,"zscore":0}
        df_sorted = df.sort_values(col, ascending=not maximize).reset_index(drop=True)
        top = df_sorted.iloc[0]
        value = float(top[col])
        prob = float(top['softmax_score'])
        ev = float(top['expected_value'])
        odds = float(top['オッズ'])

        # 2番目との差
        if len(df_sorted) > 1:
            diff2 = float(value - df_sorted.iloc[1][col])
        else:
            diff2 = value

        # Zスコア（平均との差 / 標準偏差）
        mean_val = float(df[col].mean())
        std_val = float(df[col].std() + 1e-9)
        zscore = float((value - mean_val) / std_val)

        return {"prob":prob,"ev":ev,"odds":odds,"value":value,"diff2":diff2,"zscore":zscore}

    # --- 各戦略トップ馬 ---
    softmax_top = get_top_features(race_df, "softmax_score", maximize=True)
    ev_top      = get_top_features(race_df, "expected_value", maximize=True)
    last3f_top  = get_top_features(race_df, "av後3F", maximize=False)   # 小さい方が良い
    speed_top   = get_top_features(race_df, "avスピード指数", maximize=True)
    uptrend_top = get_top_features(race_df, "上昇度", maximize=True)

    # --- コンテキストベクトル作成 ---
    context = np.array([
        n_horses,
        odds_std, fav_odds, fav_prob, longshot_ratio,

        # 各トップ馬の summary (prob, ev, odds)
        softmax_top["prob"], softmax_top["ev"], softmax_top["odds"],
        ev_top["prob"], ev_top["ev"], ev_top["odds"],
        last3f_top["prob"], last3f_top["ev"], last3f_top["odds"],
        speed_top["prob"], speed_top["ev"], speed_top["odds"],
        uptrend_top["prob"], uptrend_top["ev"], uptrend_top["odds"],

        # 飛び抜け度 (diff2, zscore)
        softmax_top["diff2"], softmax_top["zscore"],
        ev_top["diff2"], ev_top["zscore"],
        last3f_top["diff2"], last3f_top["zscore"],
        speed_top["diff2"], speed_top["zscore"],
        uptrend_top["diff2"], uptrend_top["zscore"],
    ], dtype=float)

    return context



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

def calc_reward_smooth(chosen_row, df_payout, stake=100, max_rel=3):
    # 着順に応じて滑らかに報酬を付与
    try:
        rank = chosen_row['着順']
        if rank == 1:
            rel = max_rel
        elif rank == 2:
            rel = max(max_rel - 1, 0)
        elif rank == 3:
            rel = max(max_rel - 2, 0)
        else:
            rel = max_rel / rank
        reward = rel * chosen_row['softmax_score']  # スコアも加味
        reward /= max_rel  # 0-1スケール
    except Exception:
        reward = 0.0
    return reward

def calc_reward_ev_norm(chosen_row, df_payout, stake=100, bet_type="単勝", max_ev=10):
    race_id = chosen_row['レースID']
    horse_no = chosen_row['馬番']
    try:
        payout_row = df_payout.loc[(race_id, horse_no)]
        ev = payout_row['オッズ'] * chosen_row['softmax_score']  # スコアとオッズの掛け算
        # max_evで割って0-1スケールに正規化
        reward = min(ev / max_ev, 1.0)
    except Exception:
        reward = 0.0
    return reward

# -----------------------
# 5) バンディットを使ったオフラインシミュレーション（1レースずつ順に回す）
# -----------------------
def run_contextual_bandit_simulation(
    df, df_payout,
    policies,  # list of (name, function, policy_kwargs)
    context_dim=30,
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

