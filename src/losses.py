import torch
import torch.nn.functional as F
import numpy as np

# LambdaRank NDCG@k 損失: ペアワイズにNDCG差で重み付け
def lambdarank_loss_at_k(preds, labels, k=3):
    n = preds.shape[0]
    # 全ペアの予測スコア差 → sigmoid で勝率確率
    diff = preds.unsqueeze(0) - preds.unsqueeze(1)
    pred_prob = torch.sigmoid(diff)
    # 正解ラベルの大小関係 (i > j なら 1)
    target = (labels.unsqueeze(0) > labels.unsqueeze(1)).float()

    # DCG 利得・割引
    gain = torch.pow(2.0, labels.float()) - 1.0
    _, rank_order = torch.sort(preds, descending=True)
    rank_pos = torch.argsort(rank_order).float() + 1.0
    discount = 1.0 / torch.log2(rank_pos + 1.0)
    # ペアを入れ替えたときのNDCG変化量
    delta_ndcg = torch.abs((gain.unsqueeze(0) * discount.unsqueeze(0) - gain.unsqueeze(1) * discount.unsqueeze(1)))

    # top-k にのみ注目
    mask_k = (rank_pos <= k).float()
    weight_k = mask_k.unsqueeze(0) * mask_k.unsqueeze(1)
    delta_ndcg = delta_ndcg * weight_k

    mask = torch.ones_like(target, dtype=torch.bool)
    mask.fill_diagonal_(False)
    bce = F.binary_cross_entropy(pred_prob[mask], target[mask], reduction='none')
    loss = torch.mean(delta_ndcg[mask] * bce)
    return loss

# ListNet 損失: softmax 確率分布の KL 距離
def listnet_loss(pred_scores, true_ranks):
    y_true = -true_ranks.float()            # 着順（小さいほど良い）→ 符号反転で大きいほど良い
    P_y = torch.softmax(y_true, dim=0)      # 正解分布
    P_z = torch.softmax(pred_scores, dim=0) # 予測分布
    loss = -(P_y * torch.log(P_z + 1e-12)).sum()  # クロスエントロピー
    return loss

# 期待値損失: 予測確率×オッズの期待値と実際の配当の差
def safe_ev_loss(preds, odds, is_win, clip_logits=10.0, clip_odds=50.0):
    preds = torch.clamp(preds, -clip_logits, clip_logits)
    odds = torch.log1p(odds)                # 対数オッズ（裾を抑える）
    odds = torch.clamp(odds, 1.0, clip_odds)
    probs = F.softmax(preds, dim=0)         # 予測確率
    ev_true = torch.sum(is_win * odds)      # 実際の配当期待値
    ev_pred = torch.sum(probs * odds)       # 予測配当期待値
    loss = torch.abs(ev_true - ev_pred)     # 絶対差を最小化
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.tensor(0.0, device=preds.device)
    return loss

# 強化学習（REINFORCE）: ソフトマックス方策でROI最大化
def roi_policy_loss(preds, odds, is_win, temperature=1.0):
    probs = F.softmax(preds / temperature, dim=0)  # 方策確率
    reward = is_win * odds - 1.0                    # 報酬 = 払戻 - 1
    baseline = torch.sum(probs * reward)            # ベースライン（分散低減）
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(probs + 1e-9) * advantage)  # 方策勾配
    return loss

# シグモイド版REINFORCE: 各馬独立に買う/買わないの確率で方策勾配
def roi_policy_sigmoid_loss(preds, odds, is_win, temperature=1.0):
    buy_prob = torch.sigmoid(preds / temperature)   # 買う確率
    reward = is_win * odds - 1.0
    baseline = torch.sum(buy_prob * reward) / buy_prob.sum().clamp(min=1.0)
    advantage = reward - baseline.detach()
    loss = -torch.sum(torch.log(buy_prob + 1e-9) * advantage)
    return loss
