# プロジェクト進捗ログ

## 2026-06-04 - listwise/predictor バグ修正・損失関数改善

### 修正内容
- `騎手_馬場_te` → `騎手_フィールド_te`（model_config.py, lightgbm_lambda_main.py）
  - TE実装は `['騎手', 'フィールド']` から `騎手_フィールド_te` を生成するが、scale_cols に `騎手_馬場_te` と書いてあり KeyError 原因に
- `add_history_features` から不要な `cfg.extend` 3行を削除（features.py）
  - `build_dataset.py` が既にCSVに履歴列を含めているため、`feature_cols`/`scale_cols`/`feature_category` への登録は不要で副作用のみ
- predictor.py に `add_history_features` 呼び出しを追加、log2.txt廃止・print表示に変更
- 損失関数改善（losses.py）: `squared_hinge`+`margin=0`+`ev_i` → `logistic`+`margin=0`+`value_i`
  - 従来は全馬同じスコアに収束して勾配消失していた
  - `logistic` は softplus(-diff) で常に勾配が残るため安定
- `src/lightgbm/__init__.py` 削除（不要なパッケージ化）

### 変更ファイル
- `src/listwise/model_config.py` (カラム名修正)
- `src/lightgbm_lambda_main.py` (カラム名修正)
- `src/listwise/features.py` (不要なcfg.extend削除)
- `betting/predictor.py` (add_history_features呼び出し追加, log出力→print)
- `src/listwise/losses.py` (pairwise/logistic, weight_mode/value_i)
- `src/lightgbm/__init__.py` (削除)
- `README.md` (新規作成・ポートフォリオ用)

## 2026-06-03 - 新規6特徴量を実装 + lightgbm_lambda_main対応

### 実装した特徴量（`src/listwise/features.py` に全関数追加）

| # | 特徴量 | 関数 | 優先度 |
|---|--------|------|--------|
| 1 | 距離×父馬TE | `add_interaction_te_fold(['距離グループ','父馬'])` | 高 |
| 2 | 騎手×距離TE | `add_interaction_te_fold(['騎手','距離'])` | 高 |
| 3 | 騎手×馬場TE | `add_interaction_te_fold(['騎手','フィールド'])` | 中 |
| 4 | 出走間隔クラス | `add_interval_class()` | 中 |
| 5 | 上がり3Fレース内ランク | `add_last3f_race_rank()` | 中 |
| 6 | 馬体重トレンド傾き | `add_weight_trend_slope()` | 低 |

### 変更ファイル

- **`src/listwise/features.py`**: 全6関数を追加。`add_interaction_te_fold` は内部5-fold CVでリーク防止
- **`src/listwise/model_config.py`**: `scale_cols` に新規6列追加、`common_cols` に `距離グループ` 追加
- **`src/listwise_main.py`**: fold前に非TE特徴量(4,5,6)計算、fold内に交互作用TE(1,2,3)計算を追加
- **`src/lightgbm_lambda_main.py`**: 同様の変更（メイン開発中のため）
- **`AGENTS.md`**: 新規作成（セッション切断対策）
- **`.clinerule`**: 永続化ルールを追加（起動時にAGENTS.md自動読み込み）

### 距離グループ定義
- 短距離(0): ~1400m
- マイル(1): 1401~1800m
- 中距離(2): 1801~2400m
- 長距離(3): 2401m~

### 間隔クラス定義
- 連闘(0): 0週
- 中1週(1): 1~2週
- 中2週(2): 3~4週
- 休み明け(3): 5~8週
- 長期休養(4): 9週~

### テスト
- import/syntax 確認済み
- サンプルデータでの動作確認済み（全6特徴量）

### 次のステップ候補
- 実際に学習を回して特徴量重要度を確認
- 効果が薄い特徴量の削除・調整
- 他の競馬場（東京・阪神）への展開

---

## 2026-06-03 - lightgbm_lambda_main 5-Fold 学習結果（新規6特徴量あり）

### ワイド top2 ROI 5-Fold 比較

| Signal | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | **平均** | レンジ |
|--------|-------|-------|-------|-------|-------|---------|-------|
| wide_benter_all | 82.5% | 82.2% | 83.1% | 83.0% | 76.8% | **81.5%** | 6.4% |
| wide_benter_free | 90.5% | 80.4% | 78.4% | 84.7% | 74.3% | **81.7%** | 16.2% |
| wide_benter_t2 | 78.7% | 78.2% | 76.6% | 81.5% | 83.1% | **79.6%** | 6.5% |
| **wide_benter_t2f** | **79.5%** | **82.2%** | **86.4%** | **82.7%** | **79.8%** | **82.1%** | **6.9%** |
| wide_benter_blend | 80.8% | 82.1% | 78.7% | 86.2% | 81.7% | **81.9%** | 7.5% |
| wide_hybrid_z | 90.2% | 81.1% | 77.6% | 80.1% | 81.5% | **82.1%** | 12.6% |
| wide_pred_all | 76.0% | 73.1% | 74.4% | 84.8% | 84.9% | **78.6%** | 11.8% |

### 分析サマリ

**top2 odds-free (wide_benter_t2f) が平均82.1%、レンジ6.9%で最良。** 安定性と平均値のバランスが最も良い。

| 指標 | 値 | 判断 |
|------|-----|------|
| 単勝的中率 | 23~26% | 安定的（fold間で安定） |
| 単勝回収率 | 71~83% | 全foldで100%未満だが安定 |
| `model≠人気1` 回収率 | 72~90% | 穴推奨として悪くないが的中率15~17% |
| `model=人気1` 回収率 | 69~81% | 堅実だが回収率は低め |
| val wide roi | 77~103% | fold2のみ103%と跳ねる（過学習の可能性） |

### 新旧比較（前回提示 vs 今回）

| Signal | 旧（特徴量なし？） | 新（6特徴量あり） | 変化 |
|--------|-------------------|-------------------|------|
| wide_benter_t2 | 84.3% (range 14.2%) | 79.6% (range 6.5%) | avg▼4.7%, 安定性▲ |
| wide_benter_t2f | 82.2% (range 13.7%) | 82.1% (range 6.9%) | avg横ばい, 安定性大幅▲ |
| wide_benter_free | 82.1% (range 7.0%) | 81.7% (range 16.2%) | avg横ばい, 安定性▼ |
| wide_benter_all | 81.7% (range 14.7%) | 81.5% (range 6.4%) | avg横ばい, 安定性▲ |

**考察：新特徴量を入れるとFold間の安定性が全体的に向上した（range縮小）が、平均ROIはやや低下傾向。** 特に以下の点が注目：

1. **wide_benter_t2 が79.6%に低下** → top2分類器(full)は新特徴量のノイズに弱い可能性。過学習気味だったのが正則化されたとも言える
2. **wide_benter_t2f は82.1%を維持しつつrange 13.7%→6.9%に半減** → odds-free版は新特徴量と相性が良い
3. **wide_benter_free のrangeが7.0%→16.2%に悪化** → win分類器(odds-free)は新特徴量の影響を不安定に受ける

### 所感
- **wide_benter_t2f が最安定。** 新特徴量追加による汎化性能向上が odds-free top2 に最も明確に出ている
- t2fがt2を上回り続けているのは、odds-freeの正則化効果 + 新特徴量が市場オッズとは独立した情報を提供しているから
- wide_hybrid_z (Fold1で90.2%) は新特徴量下でも跳ねるが不安定。ブレンド用シグナルとしての位置づけが妥当

---

## 2026-06-03 - Benter較正改善: α,βをfold間中央値に固定

### 変更内容
`src/lightgbm_lambda_main.py` に以下の変更を追加：

1. **α,β collector 追加（foldループ内）**
   - 4種類のBenter較正（win, win_free, top2, top2f）それぞれの最適α,βをfoldごとに収集
   - `benter_params` dict に格納

2. **中央値再評価セクション追加（foldループ後）**
   - 全5foldのα,β中央値を計算し表示
   - 保存済み test CSV を再読み込みし、中央値α,βでBenter確率を再計算
   - ワイド top2 ROI を fold 別・平均で表示

### 出力例（次回実行時）
```
Benter Calibration: Per-fold α,β vs Fixed Median α,β
  benter_prob (win classifier)                  per-fold: (1.5,2.0), (3.0,1.5), (2.0,3.0), (4.5,0.5), (2.5,2.0)
                                               median:   α=2.5, β=2.0
  ...
Re-evaluation with median α,β
  Signal                           Fold1   Fold2   Fold3   Fold4   Fold5   平均
  ------------------------------ ------- ------- ------- ------- ------- -------
  benter_prob_t2f (odds-free top2)  ...     ...     ...     ...     ...    XX.X%
```

### 次回実行結果（中央値α,β 再評価）
```
Benter Calibration: Per-fold α,β vs Fixed Median α,β
  benter_prob (win classifier)                   per-fold: (3.5,1.0),(4.5,1.0),(4.5,5.0),(3.0,0.5),(4.5,0.5)
                                                median:   α=4.5, β=1.0
  benter_prob_free (odds-free win)               per-fold: (3.5,1.5),(3.5,3.0),(4.5,1.0),(3.5,1.0),(5.0,0.5)
                                                median:   α=3.5, β=1.0
  benter_prob_t2 (top2 classifier)               per-fold: (4.0,1.5),(2.5,4.5),(4.0,0.5),(4.0,2.5),(3.0,0.5)
                                                median:   α=4.0, β=1.5
  benter_prob_t2f (odds-free top2)               per-fold: (1.5,1.0),(2.5,3.0),(5.0,0.5),(1.5,1.0),(4.5,2.0)
                                                median:   α=2.5, β=1.0

Re-evaluation with median α,β (ワイドtop2 ROI):
  Signal                              Fold1  Fold2  Fold3  Fold4  Fold5   平均
  benter_prob (win classifier)        86.9%  82.2%  83.8%  82.5%  82.4%  83.6%
  benter_prob_free (odds-free win)    85.8%  82.6%  80.4%  84.7%  84.6%  83.6%
  benter_prob_t2 (top2 classifier)    78.7%  80.1%  76.8%  79.0%  80.0%  78.9%
  benter_prob_t2f (odds-free top2)    80.8%  80.1%  83.7%  83.8%  80.8%  81.8%
```

### 新特徴量 重要度（LambdaRank gain top15）
| 順位 | 特徴量 | スコア | 備考 |
|------|--------|-------|------|
| 1 | 1人気 | 22633 | 市場 |
| 2 | 父馬 | 20910 | TE |
| 3 | 騎手 | 20464 | TE |
| 11 | **前走後3F_レース内順位** | 1203 | **新#5 健闘** |
| 13 | **騎手_距離_te** | 886 | **新#2 まずまず** |
| 22 | **騎手_フィールド_te** | 520 | **新#3 中程度** |

### 分析
**中央値α,βの効果**:
- win分類器系（benter/benter_free）がper-fold 81.5%→83.6%に改善 (+2.1%)
- レンジも6.4%→4.7%に縮小＝安定性向上
- t2fは変化ほぼなし（82.1%→81.8%）＝元々α,βにロバスト
- **新たな最良シグナル**: benter_prob / benter_prob_free（中央値）が**83.6%**でt2fを逆転

### 次回以降の改善候補（優先順）
1. **中央値α,βをデフォルト採用** (高) — per-fold grid searchはval過適合。中央値固定が安定して高ROI
2. **中央値α,β同士のEnsemble** (高) — benter_prob(83.6%) + benter_prob_free(83.6%) を加重平均
3. **特徴量重要度の低いものを削除** (中) — `間隔クラス`, `馬体重_trend_slope`, `距離グループ_父馬_te` がtop30未満
4. **α,β grid search の細分化** (低) — step 0.5→0.25、範囲も中央値周辺に絞る
5. **ハイパーパラメータチューニング** (低) — optunaでlearning_rate/num_leaves/lambda等
