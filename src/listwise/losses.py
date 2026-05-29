import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# ユーティリティ
# =============================================================================

def rank_based_topk_p_model(preds, k=3, tau=1.0):
    """
    予測スコアから、上位 k 頭のみに確率を集中させた分布を生成する。
    市場分布との乖離（ディストーション）の計算に使う。
    """
    ranks = torch.argsort(torch.argsort(-preds))
    scores = -ranks.float()
    topk = scores >= -k
    scores = torch.where(topk, scores, torch.tensor(-1e9, device=preds.device))
    return torch.softmax(scores / tau, dim=0)


def market_distortion_score(pred_scores, odds, mask=None, eps=1e-8):
    """
    モデルの予測分布（top-k）と市場分布（オッズ逆数）の
    KL ダイバージェンス × EV 重みを計算する。
    値が大きいほど「モデルが市場と乖離した予想をしている」＝ 歪み大。
    """
    if pred_scores.dim() == 1:
        pred_scores = pred_scores.unsqueeze(0)
        odds = odds.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    p_model = rank_based_topk_p_model(pred_scores)

    inv_odds = 1.0 / (odds + eps)
    if mask is not None:
        inv_odds = inv_odds * mask
    inv_odds = inv_odds * (p_model > 0).float()
    p_mkt = inv_odds / (inv_odds.sum(dim=1, keepdim=True) + eps)

    kl = p_model * torch.log((p_model + eps) / (p_mkt + eps))
    ev = torch.clamp(odds * p_model - 1.0, max=5.0)
    ev_weight = torch.relu(ev)

    distortion = (kl * ev_weight).sum(dim=1)
    return distortion.mean()


def distortion_gate(z, z0=0.2, sharpness=3.0):
    """
    ディストーション z 値を [0,1] のゲート値に変換する。
    z が z0 を超えるとゲートが開き（1に近づく）、
    「市場と乖離した予想＝賭ける価値あり」と判断する。
    """
    if not torch.is_tensor(z):
        z = torch.tensor(z)
    return torch.sigmoid((z - z0) * sharpness)


def roi_adjusted_score(preds, odds, alpha=0.1):
    """
    推論時にオッズで予測スコアを補正する後処理用。
    オッズが高い馬のスコアを α 倍だけ底上げする。
    """
    roi_factor = torch.log1p(odds)
    adjusted = preds * (1 + alpha * roi_factor)
    return adjusted


# =============================================================================
# ランク損失（ListNet・LambdaRank）
# =============================================================================

def listnet_loss(pred_scores, true_ranks):
    """
    ListNet 損失: 着順を softmax 確率分布とみなし、
    予測分布とのクロスエントロピーを最小化する。
    着順が良い（数字が小さい）ほど大きな確率を与える。
    """
    y_true = -true_ranks.float()
    P_y = torch.softmax(y_true, dim=0)
    P_z = torch.softmax(pred_scores, dim=0)
    loss = -(P_y * torch.log(P_z + 1e-12)).sum()
    return loss


def lambdarank_loss_at_k(preds, labels, k=3):
    """
    LambdaRank NDCG@k 損失: 馬i > 馬j の入れ替えが NDCG@k に
    与える変化量 ΔNDCG で各ペアの BCE 損失を重み付けする。
    top-k にランクインする馬の順位を正確にするよう学習する。
    """
    n = preds.shape[0]
    diff = preds.unsqueeze(0) - preds.unsqueeze(1)
    pred_prob = torch.sigmoid(diff)
    target = (labels.unsqueeze(0) > labels.unsqueeze(1)).float()

    gain = torch.pow(2.0, labels.float()) - 1.0
    _, rank_order = torch.sort(preds, descending=True)
    rank_pos = torch.argsort(rank_order).float() + 1.0
    discount = 1.0 / torch.log2(rank_pos + 1.0)
    delta_ndcg = torch.abs((gain.unsqueeze(0) * discount.unsqueeze(0) - gain.unsqueeze(1) * discount.unsqueeze(1)))

    mask_k = (rank_pos <= k).float()
    weight_k = mask_k.unsqueeze(0) * mask_k.unsqueeze(1)
    delta_ndcg = delta_ndcg * weight_k

    mask = torch.ones_like(target, dtype=torch.bool)
    mask.fill_diagonal_(False)
    bce = F.binary_cross_entropy(pred_prob[mask], target[mask], reduction='none')
    loss = torch.mean(delta_ndcg[mask] * bce)
    return loss


# =============================================================================
# ROI / 期待値 損失
# =============================================================================

def safe_ev_loss(preds, odds, is_win, clip_logits=10.0, clip_odds=50.0):
    """
    期待値損失: ソフトマックス予測確率と勝ち馬の配当から計算した
    期待値が実際の配当と一致するよう学習する。
    極端な勾配を防ぐため logits / odds をクリップしている。
    """
    preds = torch.clamp(preds, -clip_logits, clip_logits)
    odds = torch.log1p(odds)
    odds = torch.clamp(odds, 1.0, clip_odds)
    probs = F.softmax(preds, dim=0)
    ev_true = torch.sum(is_win * odds)
    ev_pred = torch.sum(probs * odds)
    loss = torch.abs(ev_true - ev_pred)
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=preds.device)
    return loss


def ev_huber_loss(pred, odds, is_win, delta=1.0):
    """
    Huber 版 EV 損失: 誤差が delta 以内なら二乗、超えたら線形に切り替え。
    外れ値にロバストな EV 学習を行う。
    """
    target = is_win * odds - 1
    diff = pred - target
    abs_diff = diff.abs()
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic + 0.5 * delta**2
    return torch.where(abs_diff < delta, quadratic**2 * 0.5, linear).mean()


def expected_value_loss(preds, odds, is_win):
    """
    クロスエントロピーで勝率を学習しつつ、EV を最大化する。
    loss = CE - 0.1 * EV で、EV 項が「高いオッズの馬を買う」方向に引っ張る。
    """
    p = torch.softmax(preds, dim=0)
    ce = torch.mean(-is_win * torch.log(p + 1e-9))
    ev = torch.sum(p * odds) - 1
    return ce - 0.1 * ev


def roi_weighted_loss(preds, odds, is_win):
    """
    relu(preds) で重み付けした ROI を直接最大化する。
    重みはソフトマックスではなく relu の割合なので、
    予測値が正の馬だけに資金配分するイメージ。
    """
    weights = torch.relu(preds) / torch.sum(torch.relu(preds) + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(weights * reward)
    return -roi


def weighted_roi_loss(preds, odds, is_win):
    """
    roi_weighted_loss に加え、低オッズ（7倍未満）の馬に2倍の重みを掛ける。
    人気馬（オッズ低）をより重視した戦略。
    """
    weights_odds = torch.where(odds < 7, 2.0, 1.0)
    weights = torch.relu(preds) / (torch.sum(torch.relu(preds)) + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(weights * reward * weights_odds)
    return -roi


def roi_with_calibration_loss(preds, odds, is_win, alpha=0.2):
    """
    ROI 最大化と Brier スコア（確率較正）のトレードオフ。
    ROI -= α * Brier で、予測確率が実際の勝率と
    乖離しすぎないようにペナルティを課す。
    """
    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(w * reward)
    probs = torch.softmax(preds, dim=0)
    brier = torch.mean((probs - is_win)**2)
    return -(roi - alpha * brier)


def sharpe_roi_loss(preds, odds, is_win):
    """
    シャープレシオ（ROI / 標準偏差）を最大化する。
    単なる ROI ではなく、リスク（バラツキ）を考慮した
    安定した収益を目指す。
    """
    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)
    reward = is_win * odds - 1
    roi = torch.sum(w * reward)
    variance = torch.sum(w * (reward - roi)**2)
    sharpe = roi / torch.sqrt(variance + 1e-9)
    return -sharpe


def differentiable_roi_loss(preds, odds, is_win, threshold=1.0, beta=10.0):
    """
    「買う/買わない」をシグモイドで緩和し、購入確率 × リターンの
    期待値を最大化する。beta が大きいほど買う/買わないの閾値判定が
    明確になる。
    """
    probs = torch.sigmoid(preds)
    buy_prob = torch.sigmoid(beta * (probs * odds - threshold))
    roi = torch.sum(buy_prob * (is_win * odds - 1.0))
    return -roi


def roi_policy_loss(preds, odds, is_win, temperature=1.0):
    """
    REINFORCE（強化学習）: ソフトマックス方策に従い、
    報酬（払戻-1）の期待値を最大化する。
    ベースラインで分散を低減している。
    """
    probs = F.softmax(preds / temperature, dim=0)
    reward = is_win * odds - 1.0
    baseline = torch.sum(probs * reward)
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(probs + 1e-9) * advantage)
    return loss


def roi_policy_sigmoid_loss(preds, odds, is_win, temperature=1.0):
    """
    シグモイド版 REINFORCE: 各馬を独立に「買う/買わない」の
    二値方策で学習する。ソフトマックス版と異なり、
    他馬の確率とトレードオフしない。
    """
    buy_prob = torch.sigmoid(preds / temperature)
    reward = is_win * odds - 1.0
    baseline = torch.sum(buy_prob * reward) / buy_prob.sum().clamp(min=1.0)
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(buy_prob + 1e-9) * advantage)
    return loss


def combined_roi_loss(preds, odds, is_win, alpha=0.5):
    """roi_policy_loss と differentiable_roi_loss を α:1-α で混合"""
    return alpha * roi_policy_loss(preds, odds, is_win) \
         + (1 - alpha) * differentiable_roi_loss(preds, odds, is_win)


# =============================================================================
# ペアワイズ損失 + ROI
# =============================================================================

def pairwise_roi_loss(preds, odds, is_win, margin=0.0):
    """
    勝ち馬と各負け馬のスコア差に、ROI差をマージンとして与える。
    勝ち馬のROIが高いほど大きいマージンを要求する。
    加えて relu 重み付き ROI 項を引く。
    """
    win_idx = is_win.argmax()
    win_pred = preds[win_idx]
    win_roi = odds[win_idx] - 1

    loss = 0
    cnt = 0
    for i in range(len(preds)):
        if i == win_idx:
            continue
        diff = (win_pred - preds[i])
        target_margin = max(0, win_roi - (odds[i] - 1))
        loss += torch.relu(target_margin - diff)
        cnt += 1

    w = torch.relu(preds)
    w = w / (w.sum() + 1e-9)
    roi = torch.sum(w * (is_win * odds - 1))
    return loss / cnt - roi


def utility_aware_ranking_loss_roi(
    preds,
    values,
    payouts,
    mask=None,
    margin=0.05,
    pairwise='hinge',
    weight_mode='value_i',
    normalize_values=False,
    eps=1e-8
):
    """
    効用（value）× 配当（payout）でペアワイズ損失を重み付けする汎用損失。

    【pairwise（ペアワイズ損失の種類）】
      - 'hinge'       : max(0, margin - diff)  SVM的なマージン損失
      - 'squared_hinge': hinge の二乗。外れ値に強くなる
      - 'logistic'    : softplus(-diff)  ロジスティック回帰的に滑らか
      - 'bpr'         : -log(sigmoid(diff))  Bayesian Personalized Ranking
      - 'exp'         : exp(-diff)  指数関数的に差を拡大
      - 'soft_margin' : softplus(margin - diff)  マージン付き滑らか版
      - 'tanh'        : 1 - tanh(diff)  差が十分あれば損失0に近づく

    【weight_mode（ペアの重み付け方法）】
      - 'value_i'      : 馬iの効用値（上位馬ほど強く学習）
      - 'roi'          : 期待ROIの差（馬i - 馬j）で重み付け
      - 'abs_value_diff': 効用値の絶対差
      - 'ev_i'         : 馬iの期待値（value × payout）そのまま
      - 'softmax_ev'   : EV を softmax した確率で重み付け
      - 'rank_focus'   : 予測上位3頭のみに重みを集中
      - 'focal_roi'    : ROI差 × (1 - sigmoid(diff))^2  判定が難しいペアを重視
      - 'odds_aware'   : EV / odds でオッズ正規化
    """
    if preds.dim() == 1:
        preds = preds.unsqueeze(0)
        values = values.unsqueeze(0)
        payouts = payouts.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, N = preds.shape
    device = preds.device

    if mask is None:
        mask = torch.ones_like(preds)
    mask = mask.float()

    if normalize_values:
        val_sum = (values * mask).sum(dim=1)
        cnt = mask.sum(dim=1).clamp_min(1.0)
        val_mean = val_sum / cnt
        centered = values - val_mean.unsqueeze(1)
        mad = (centered.abs() * mask).sum(dim=1) / cnt
        mad = mad.clamp_min(1.0)
        values = centered / mad.unsqueeze(1)
        values = torch.clamp(values, -10.0, 10.0)

    pred_i = preds.unsqueeze(2)
    pred_j = preds.unsqueeze(1)
    diff = pred_i - pred_j

    val_i = values.unsqueeze(2)
    val_j = values.unsqueeze(1)
    payout_i = payouts.unsqueeze(2)
    payout_j = payouts.unsqueeze(1)

    mask_i = mask.unsqueeze(2)
    mask_j = mask.unsqueeze(1)
    pair_mask = mask_i * mask_j
    diag = torch.eye(N, device=device).unsqueeze(0)
    pair_mask = pair_mask * (1 - diag)

    if weight_mode == 'value_i':
        w = F.relu(val_i)
    elif weight_mode == 'roi':
        expected_i = val_i * payout_i
        expected_j = val_j * payout_j
        w = F.relu(expected_i - expected_j)
    elif weight_mode == 'abs_value_diff':
        w = (val_i - val_j).abs()
    elif weight_mode == 'ev_i':
        w = F.relu(val_i * payout_i)
    elif weight_mode == 'softmax_ev':
        ev = values * payouts
        sm = torch.softmax(ev, dim=1)
        w = sm.unsqueeze(2)
    elif weight_mode == 'rank_focus':
        K = min(3, N)
        topk = torch.topk(preds, K, dim=1).indices
        focus = torch.zeros_like(preds)
        focus.scatter_(1, topk, 1.0)
        w = focus.unsqueeze(2)
    elif weight_mode == 'focal_roi':
        ev_diff = (val_i * payout_i) - (val_j * payout_j)
        base = F.relu(ev_diff)
        focal = (1.0 - torch.sigmoid(diff)).pow(2)
        w = base * focal
    elif weight_mode == 'odds_aware':
        w = F.relu((val_i * payout_i) / (payout_i + eps))
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    weights = w * pair_mask
    sum_w = weights.sum(dim=(1, 2))
    zero_mask = (sum_w < eps)
    if zero_mask.any():
        weights[zero_mask] = pair_mask[zero_mask]

    if pairwise == 'hinge':
        pair_loss = F.relu(margin - diff)
    elif pairwise == 'squared_hinge':
        pair_loss = F.relu(margin - diff) ** 2
    elif pairwise == 'logistic':
        pair_loss = F.softplus(-diff)
    elif pairwise == 'bpr':
        pair_loss = -F.logsigmoid(diff)
    elif pairwise == 'exp':
        pair_loss = torch.exp(-diff)
    elif pairwise == 'soft_margin':
        pair_loss = F.softplus(margin - diff)
    elif pairwise == 'tanh':
        pair_loss = 1.0 - torch.tanh(diff)
    else:
        raise ValueError(f"Unknown pairwise: {pairwise}")

    weighted = pair_loss * weights
    sum_loss = weighted.sum(dim=(1, 2))
    sum_weights = weights.sum(dim=(1, 2)).clamp_min(eps)
    return (sum_loss / sum_weights).mean()


def combined_loss(preds, labels, odds, is_win, pairwise, weight_mode, k=3, alpha=0.05):
    """
    ランク損失 + ROI 損失の混合用ラッパー。
    現在は listnet_loss のみ返すが、コメントアウト部に
    様々な組み合わせパターンの実験跡が残っている。
    """
    loss_rank = listnet_loss(preds, labels)
    return loss_rank


# =============================================================================
# 複勝（place）系損失
# =============================================================================

def place_listnet_loss(logits, is_in, top_k=3):
    """
    複勝（3着以内）の ListNet 損失。
    正解ラベル（is_in）を確率分布とみなし、softmax 予測との
    クロスエントロピーを最小化する。
    """
    y_true = is_in.float()
    y_true = y_true / y_true.sum()
    y_pred = torch.softmax(logits, dim=0)
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-8))
    return loss


def place_ev_loss(logits, is_in, place_odds):
    """
    複勝の期待値損失。
    シグモイド確率で各馬の購入期待値を計算し、平均を最大化する。
    外れ馬にはペナルティ（1-is_in）がかかる。
    """
    place_odds_fixed = place_odds.clone()
    place_odds_fixed[place_odds_fixed == 0] = 1.0
    prob = torch.sigmoid(logits)
    reward = is_in * (place_odds_fixed - 1) - (1 - is_in)
    ev = prob * reward
    return -ev.mean()


def combined_place_loss(logits, is_in, place_odds, alpha=0, top_k=3):
    """複勝: ListNet 損失と EV 損失を α で混合する"""
    listnet = place_listnet_loss(logits, is_in, top_k=top_k)
    ev = place_ev_loss(logits, is_in, place_odds)
    return alpha * listnet + (1 - alpha) * ev


def place_contrastive_loss(logits, is_in, temperature=0.07):
    """
    複勝の対照損失。正例（複勝圏内）と負例（圏外）を
    temperature でスケーリングした softmax 交差エントロピーで学習する。
    正例のスコアが負例より相対的に高くなるよう促す。
    """
    pos = logits[is_in == 1]
    neg = logits[is_in == 0]
    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)
    loss_list = []
    for p in pos:
        scores = torch.cat([p.unsqueeze(0), neg]) / temperature
        lse = torch.logsumexp(scores, dim=0)
        log_num = p / temperature
        loss_list.append(-(log_num - lse))
    return torch.mean(torch.stack(loss_list))


def topk_place_rank_loss(logits, is_in, k=3, margin=1.0):
    """
    複勝の top-k ランキング損失。
    正例と、負例のうちスコア上位 k 頭だけを相手にマージン損失を計算する。
    不要な負例との比較を省き、効率的に学習する。
    """
    pos = logits[is_in == 1]
    neg = logits[is_in == 0]
    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)
    top_neg = torch.topk(neg, k=min(k, len(neg))).values
    loss = 0
    count = 0
    for p in pos:
        for n in top_neg:
            loss += torch.relu(margin - (p - n))
            count += 1
    return loss / count


def smooth_place_listnet_loss(logits, is_in, eps=0.1):
    """
    ラベルスムージング付き複勝 ListNet。
    正例に (1-eps)/k、負例に eps/(N-k) のソフトラベルを与える。
    過学習を抑え、汎化性能を高める効果がある。
    """
    k = is_in.sum()
    if k == 0:
        return torch.tensor(0.0, device=logits.device)
    N = len(logits)
    y_true = torch.zeros(N, device=logits.device)
    y_true[is_in == 1] = (1 - eps) / k
    y_true[is_in == 0] = eps / (N - k)
    y_pred = torch.softmax(logits, dim=0)
    loss = -torch.sum(y_true * torch.log(y_pred + 1e-8))
    return loss


def topk_place_rank_loss_roi(logits, is_in, payout, bet=None, k=1, margin=1.0, alpha=0.1, eps=1e-9):
    """
    複勝の ROI 感度調整可能な top-k ランキング損失。
    正例（複勝圏内）の ROI が高いほど重みを大きくする。
    alpha=0 で ROI 無視、alpha 大で高配当重視になる。
    """
    if bet is None:
        bet = torch.ones_like(payout) * 100
    roi = payout / (bet + eps)
    roi_weight = torch.log1p(roi) ** alpha

    pos_idx = (is_in == 1)
    pos = logits[pos_idx]
    pos_w = roi_weight[pos_idx]
    neg = logits[~pos_idx]

    if len(pos) == 0:
        return torch.tensor(0.0, device=logits.device)

    top_neg = torch.topk(neg, k=min(k, len(neg))).values
    loss = 0
    count = 0
    for p, w in zip(pos, pos_w):
        for n in top_neg:
            raw = torch.relu(margin - (p - n))
            loss += w * raw
            count += 1
    return loss / (count + eps)


def soft_topk_hit_loss(logits, is_in, k=3, tau=1.0):
    """
    複勝: softmax 予測のうち、正例の確率和を最大化する単純な損失。
    全馬の softmax 確率を計算し、複勝圏内の馬の確率を最大化する。
    シンプルだが実践的な指標。
    """
    probs = torch.softmax(logits / tau, dim=0)
    hit_prob = probs[is_in == 1].sum()
    return -hit_prob


# =============================================================================
# ゲート制御（ディストーションに基づく賭け判断）
# =============================================================================

def compute_gate(df, distortion_history, race_col="レースID", odds_col="オッズ",
                 score_col="pred_score", no_bet_threshold=0.1):
    """
    全レースの各馬に、ディストーションスコア・z値・ゲート値・
    賭け判定（bet）を計算して DataFrame に追加する。
    gate > no_bet_threshold の場合のみ賭ける。
    """
    mu = np.mean(distortion_history)
    sigma = np.std(distortion_history) + 1e-8

    df = df.copy()
    df["distortion"] = np.nan
    df["z"] = np.nan
    df["gate"] = np.nan
    df["bet"] = False

    for race_id, df_race in df.groupby(race_col):
        pred_scores = torch.tensor(df_race[score_col].values, device=device, dtype=torch.float32)
        odds = torch.tensor(df_race[odds_col].values, device=device, dtype=torch.float32)

        with torch.no_grad():
            distortion = market_distortion_score(pred_scores, odds)
            dist_z = (distortion.item() - mu) / sigma

        gate = distortion_gate(dist_z)
        gate = (gate ** 2).item()
        bet = gate > no_bet_threshold

        idx = df_race.index
        df.loc[idx, "distortion"] = distortion.item()
        df.loc[idx, "z"] = dist_z
        df.loc[idx, "gate"] = gate
        df.loc[idx, "bet"] = bet

    return df


def compute_gate_for_race(top, distortion_history, no_bet_threshold=1.0):
    """
    単一レースの top 馬に対してゲート判定を行う。
    compute_gate のレース単位版。バリデーション・テスト時に
    各レースの先頭馬にのみ適用するために使う。
    """
    with torch.no_grad():
        pred_scores = torch.tensor(top['pred_score'].values, device=device)
        odds = torch.tensor(top['オッズ'].values, device=device)
        distortion = market_distortion_score(pred_scores, odds)
        mu = np.mean(distortion_history)
        sigma = np.std(distortion_history) + 1e-8
        dist_z = (distortion.item() - mu) / sigma

    gate = distortion_gate(dist_z)
    gate = gate ** 2
    top['bet'] = gate > no_bet_threshold
    return top[top['bet'] == True]
