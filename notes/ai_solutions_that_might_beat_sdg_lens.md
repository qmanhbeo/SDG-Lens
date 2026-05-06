# AI Solutions That Could Beat SDG Lens

Focus: Real-world utility + feasible with current data/compute + higher scores.

---

## The Honest Assessment

Current SDG Lens: MiniLM on English-only subset (200 train rows), basic training.

**What could beat it**:

1. More data (full English corpus)
2. More compute (more epochs, full training)
3. Better model (BERT-base vs MiniLM)
4. Smarter training (ensemble, loss functions, thresholds)

---

## Solution A: Ensemble TF-IDF + BERT

**Why it could win**
TF-IDF is fast and captures explicit keywords. BERT captures context. Combined, they cover both signal types. Proven to beat individual models in NLP competitions.

**Feasibility**: High — reuse r_Ting TF-IDF + SDG Lens BERT

**Pipeline**
1. Train r_Ting's TF-IDF + LinearSVC (already done)
2. Train SDG Lens BERT
3. Combine predictions: weighted average of probabilities
4. Optimize weights on validation set

**Expected improvement**: +3-8% F1 over single-model BERT

**Real-world value**: More robust — TF-IDF catches obvious keywords, BERT handles nuance.

---

## Solution B: Per-SDG Threshold Optimization

**Why it could win**
SDG Lens uses global threshold (0.5). But SDGs are imbalanced: SDG 1 has way more samples than SDG 17. Per-SDG thresholds maximize F1 for each goal.

**Feasibility**: High — simple grid search per label

**Pipeline**
1. Train BERT normally
2. For each of 17 SDGs, search threshold 0.1→0.9 to maximize F1
3. Store optimal threshold per SDG
4. At inference: apply per-SDG threshold

**Expected improvement**: +5-10% macro F1 (helps rare classes most)

**Real-world value**: Fixes " SDG 17 always gets missed" problem.

---

## Solution C: Label Correlation Chaining

**Why it could win**
SDGs aren't independent. If doc has SDG 1 (Poverty), it's likely also has SDG 3 (Health), SDG 4 (Education). Model this correlation as a graph, use as prior.

**Feasibility**: Medium — requires correlation analysis + graph modification

**Pipeline**
1. Compute co-occurrence matrix from SDGi labels
2. Build label graph (SDG 1 → SDG 3 has strong edge)
3. During inference: if SDG 1 is predicted, boost probability of SDG 3
4. Or: train classifier to predict label pairs first, then refine

**Expected improvement**: +3-5% F1 on co-occurring labels

**Real-world value**: Makes predictions more coherent (no orphan SDGs).

---

## Solution D: Focal Loss for Imbalance

**Why it could win**
SDGi is 89% single-label. BCE treats all positives equally. Focal loss down-weights easy negatives, focuses on hard ones — better for multi-label.

**Feasibility**: High — one line change in loss function

**Pipeline**
```python
# Replace BCEWithLogitsLoss with FocalLoss
loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, labels)
```

**Expected improvement**: +2-5% F1 on minority classes

**Real-world value**: Better coverage of rare SDGs (SDG 14 Life Below Water, SDG 17 Partnerships).

---

## Solution E: Semi-Supervised with Embeddings

**Why it could win**
SDGi comes with pre-computed 1536-dim embeddings (text-embedding-ada-002). Use nearest-neighbor to create pseudo-labels for unlabeled data, expand training set.

**Feasibility**: High — embeddings already provided

**Pipeline**
1. For each test/unlabeled doc, find k-nearest neighbors in train (cosine similarity)
2. If neighbors agree (>80% same label), add as pseudo-labeled example
3. Retrain with expanded dataset

**Expected improvement**: +5-15% effective training size → better generalization

**Real-world value**: Leverages the provided embeddings that are currently unused.

---

## Solution F: Hierarchical Classification

**Why it could win**
17 classes is hard. First classify into 3 groups (Economic/Social/Environmental), then within group. Reduces search space at each step.

**Feasibility**: Medium — needs restructuring

**Pipeline**
1. Map 17 SDGs to 3 categories:
   - Economic: SDG 8, 9, 10, 11, 12
   - Social: SDG 1, 2, 3, 4, 5, 16, 17
   - Environmental: SDG 6, 7, 13, 14, 15
2. Train level-1 classifier (3-way)
3. Train 3 level-2 classifiers (within each group)
4. At inference: combine predictions

**Expected improvement**: +3-7% F1 (smaller decision space per classifier)

**Real-world value**: More interpretable — first says "this is about environment", then "specifically water".

---

## Solution G: Curriculum Learning

**Why it could win**
Train on "easy" examples first (high confidence, single label), then "hard" examples (rare labels, ambiguous). Model builds foundation before facing difficult cases.

**Feasibility**: Medium — need to define difficulty metric

**Pipeline**
1. Score training examples by difficulty (label frequency + prediction confidence from TF-IDF)
2. Sort by difficulty, train in phases:
   - Phase 1: top 50% easiest examples
   - Phase 2: all examples
3. Or: weighted sampling during training

**Expected improvement**: +2-5% F1, faster convergence

**Real-world value**: More stable training, better final performance.

---

## Solution H: Language-Aware Multi-Task

**Why it could win**
Predict language (EN/ES/FR) as auxiliary task alongside SDG. Forces model to learn language-agnostic representations that transfer to classification.

**Feasibility**: High — SDGi has language metadata

**Pipeline**
```python
# Multi-task loss
sdg_loss = BCEWithLogitsLoss()(sdg_logits, labels)
lang_loss = CrossEntropyLoss()(lang_logits, lang_labels)
total_loss = sdg_loss + 0.3 * lang_loss
```

**Expected improvement**: +2-4% F1, especially on non-English

**Real-world value**: Better cross-lingual generalization without explicit multilingual model.

---

## Comparison: Which Could Actually Beat SDG Lens?

| Solution | Score Impact | Feasibility | Real-World Value | Priority |
|----------|-------------|-------------|------------------|----------|
| **B. Per-SDG Threshold** | +5-10% macro | ✅ High | High (fixes rare classes) | 1 |
| **A. Ensemble** | +3-8% | ✅ High | High (robustness) | 1 |
| **E. Semi-Supervised** | +5-15% data effect | ✅ High | High (uses unused embeddings) | 1 |
| **D. Focal Loss** | +2-5% | ✅ High | Medium (rare classes) | 2 |
| **C. Label Chaining** | +3-5% | ⚠️ Medium | High (coherence) | 2 |
| **F. Hierarchical** | +3-7% | ⚠️ Medium | Medium (interpretability) | 3 |
| **H. Multi-Task** | +2-4% | ✅ High | Medium (cross-lingual) | 3 |
| **G. Curriculum** | +2-5% | ⚠️ Medium | Low (training artifact) | 4 |

**Top 3 recommendations** (high impact + feasible):
1. **Per-SDG threshold** — simplest, high impact on macro F1
2. **Ensemble** — proven, combines best of r_Ting and SDG Lens
3. **Semi-supervised** — leverages provided embeddings, likely biggest data gain

---

## What Actually Beats SDG Lens?

The honest truth: **More compute + more data beats clever architecture.**

- Full English training set (~4000 docs) vs 200 → +20-30% F1
- BERT-base vs MiniLM → +5-10% F1
- 5 epochs vs 1 → +5-10% F1

The solutions above are optimizations. The baseline beatdown is: train properly on full data with bigger model.

**Recommendation**: Combine (A) Ensemble + (B) Threshold + (E) Semi-supervised. That's the realistic winner.