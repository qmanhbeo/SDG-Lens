# Feasibility Assessment: Solution 1 (SDG Text Explainability)

**Solution**: Multi-label SDG classifier with token-level rationale extraction  
**Based on**: HateXplain attention mechanism (r_Aiden) + SDGi dataset

---

## 1. Repo Inspection Summary

### r_Ting Pipeline (TF-IDF + Linear)

| Component | File | Reusable? |
|-----------|------|----------|
| Data loading | `sdgi_replication.py:183-254` | ✅ Direct use |
| Multi-label handling | `RobustMultiLabelSGD` (lines 60-128) | ❌ Different model type |
| TF-IDF vectorizer | Lines 390-400 | ❌ Won't use (using BERT) |
| Training loop | Lines 358-487 | ❌ Different approach |
| Evaluation | `evaluate_predictions` | ✅ Can adapt |

**Conclusion**: Only utility functions (seed setting, basic evaluation) usable. The core pipeline must be rewritten for BERT.

---

### r_Aiden Pipeline (BERT + Attention)

| Component | File | Reusable? |
|-----------|------|----------|
| Model architecture | `src/model.py:21-80` | ✅ **Direct use** (key) |
| Attention extraction | `src/model.py:48-51` | ✅ **Direct use** (key) |
| Data loading | `src/data.py:88-163` | ❌ Dataset format differs |
| Training loop | `src/train.py:108-178` | ✅ Can adapt |
| Token alignment | `src/train.py:43-78` | ✅ Needs modification |
| Config | `configs/fast.yaml` | ✅ Can adapt |

**Key insight**: `BertAttentionClassifier` is domain-agnostic. Only needs `num_labels` change.

---

## 2. Code Mapping: r_Aiden ↔ SDGi

### What Can Be Reused Directly

| r_Aiden Component | SDGi Adaptation |
|------------------|------------------|
| `BertAttentionClassifier` class | Change `num_labels=17` |
| Attention extraction logic (lines 48-51) | Works unchanged |
| Rationale loss calculation (lines 61-72) | Works unchanged |
| Training loop structure | Adapt for multi-label |
| Config YAML | Update parameters |

### What Must Be Modified

| Component | Required Change |
|-----------|----------------|
| **Model** | Multi-label instead of multi-class (3 → 17, sigmoid vs softmax) |
| **Data loader** | SDGi HuggingFace dataset format |
| **Loss function** | BCEWithLogitsLoss for multi-label |
| **Labels** | 17 binary targets per sample |
| **Rationale targets** | SDGi has NO rationale annotations — must generate synthetic or use attention directly as rationale |

### r_Ting Role

**Discard for Solution 1**. BERT approach is fundamentally different. r_Ting useful only as baseline comparison (optional).

---

## 3. Implementation Plan

### Step 1: Data Preparation

| Task | Difficulty | Notes |
|------|------------|-------|
| Load SDGi from HuggingFace | Low | `load_dataset("UNDP/sdgi-corpus")` |
| Format to match r_Aiden input | Medium | Need tokens + labels + (no rationale in SDGi) |
| Create train/valid/test split | Low | Use predefined split |

**Critical**: SDGi has NO token-level rationales. Two options:
- Option A: Use attention weights as implicit rationales (no additional annotation)
- Option B: Use a simple heuristic (e.g., keyword presence) to create synthetic rationales

**Recommended**: Option A — use attention directly as rationales (simpler, avoids annotation cost)

---

### Step 2: Model Modification

| Task | Difficulty | Notes |
|------|------------|-------|
| Modify `BertAttentionClassifier` | Medium | Change to multi-label |
| Replace CrossEntropyLoss → BCEWithLogitsLoss | Low | Single line change |
| Adjust attention aggregation | Low | May need scaling for 17 labels |

Key modification in `model.py`:
```python
# Current (multi-class):
logits = self.classifier(self.dropout(pooled))
cls_loss = self.ce_loss(logits, labels)

# Needed (multi-label):
logits = self.classifier(self.dropout(pooled))  # [B, 17]
cls_loss = nn.BCEWithLogitsLoss()(logits, labels.float())
```

---

### Step 3: Training Setup

| Task | Difficulty | Notes |
|------|------------|-------|
| Adapt training loop | Low | Standard PyTorch pattern |
| Batch size adjustment | Low | SDGi texts are longer than HateXplain |
| Learning rate tuning | Medium | May need adjustment |
| Epochs | Medium | Expect 3-5 epochs |

**Expected challenges**:
- 17 labels = sparse (most samples have only 1-2 positive labels)
- Class imbalance — may need weighted loss or sampling

---

### Step 4: Evaluation

| Task | Difficulty | Notes |
|------|------------|-------|
| F1 scores (micro/macro/weight) | Low | Use sklearn |
| Rationale evaluation | Medium | No ground truth — evaluate attention quality |
| Compare with r_Ting baseline | Low | Run r_Ting for comparison |

**Rationale evaluation**: Since SDGi has no ground-truth rationales:
- Visual inspection of top-attended tokens
- Correlation with SDG keywords as proxy metric

---

### Step 5: Visualization of Explanations

| Task | Difficulty | Notes |
|------|------------|-------|
| Attention heatmaps | Low | Standard matplotlib |
| Token highlighting | Low | Map attention to words |
| UI for explanation | Medium | Streamlit app optional |

---

## 4. Time Estimation

### Assumptions
- Familiar with PyTorch + basic NLP
- 4-6 focused hours/day on implementation

### Best-Case Timeline

| Phase | Days | Cumulative |
|-------|------|----------|
| Setup (env, data loading) | 1 | 1 |
| Model modification | 1 | 2 |
| Training loop + initial run | 2 | 4 |
| Debugging + tuning | 2 | 6 |
| Evaluation + comparison | 1 | 7 |
| Visualization | 1 | 8 |
| Documentation/write-up | 2 | 10 |

**Best-case: ~10 days**

---

### Expected Timeline

| Phase | Days | Cumulative |
|-------|------|----------|
| Setup | 2 | 2 |
| Model modification | 2 | 4 |
| Training | 3 | 7 |
| Debugging | 3 | 10 |
| Evaluation | 2 | 12 |
| Visualization | 2 | 14 |
| Write-up | 3 | 17 |

**Expected: ~17 days (≈ 3.5 weeks)**

---

### Worst-Case Timeline

| Phase | Days | Cumulative |
|-------|------|----------|
| All above + issues | +7 | 24 |
| Multi-label instability | +5 | 29 |
| Unexpected GPU issues | +3 | 32 |

**Worst-case: ~32 days (~6.5 weeks)**

---

## 5. Risks & Bottlenecks

### Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Multi-label training instability** | High — sparse labels cause training issues | Use weighted BCE, adjust threshold, use focal loss |
| **Attention not meaningful for SDGs** | Medium — rationales may be noisy | Evaluate with keyword correlation, use as soft证据 only |
| **Long text truncation** | Medium — SDGi texts are long (up to 2048 tokens) | Increase max_length or use hierarchical pooling |

### Data Issues

| Risk | Impact | Mitigation |
|------|--------|------------|
| **No rationale annotations** | High — cannot evaluate explanations | Use attention directly as implicit rationale; no ground-truth evaluation |
| **Class imbalance** | Medium — 89% have single label | Use class weights, oversampling |

### Evaluation Challenges

| Risk | Impact | Mitigation |
|------|--------|------------|
| **No baseline for rationale quality** | Medium — unknown what "good" looks like | Use proxy: correlation with SDG keywords; human evaluation |

---

## 6. Verdict

### Final Assessment

| Criteria | Rating |
|----------|-------|
| **Feasible within MSc timeline?** | **Yes** |
| **Confidence level** | **High** |
| **Recommended scope** | **Full implementation** (with Option A rationales) |

### Rationale

1. ✅ Architecture directly reusable from r_Aiden
2. ✅ Only non-trivial change: multi-label (well-understood problem)
3. ✅ Clear evaluation: F1 scores + attention visualization
4. ✅ SDGi is well-structured HuggingFace dataset
5. ⚠️ No rationale ground truth → use attention as implicit explanation

### Suggested Scope

**Full implementation** recommended:
- Multi-label BERT classifier (17 labels)
- Attention-based rationale extraction
- Comparison with r_Ting baseline
- Visualization of top-attended tokens per SDG

**If time-constrained**: Skip rationale extraction, focus on multi-label classification only.

---

## Appendix: Key File References

| File | Lines | Purpose |
|------|-------|---------|
| `r_Aiden/src/model.py` | 21-80 | BertAttentionClassifier (reuse directly) |
| `r_Aiden/src/train.py` | 108-178 | Training loop (adapt) |
| `r_Ting/sdgi_replication.py` | 1-589 | SDGi loading (reference) |
| `SDGi/README.md` | 1-286 | Dataset documentation |