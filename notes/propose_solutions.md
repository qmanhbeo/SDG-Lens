# Proposed AI Solutions

Based on inspection of the repository assets (datasets + existing pipelines), this document proposes original AI solutions that build on top of existing work.

---

## Capabilities Summary

### Available Assets

| Asset | Input | Output | Strengths | Limitations |
|-------|-------|--------|-----------|--------------|
| **GSP/** (Country SDG scores) | Tabular (168 countries) | Numerical scores (0-100), clusters | Rich country-level metrics, progress tracking | No text, aggregated only |
| **SDGi/** (UNDP articles) | Multi-label text (5880 train, 1470 test) | 17 SDG labels | Multilingual (EN/FR/ES), rich metadata, embeddings | No rationale/explanation |
| **HateXplain/** (Hate speech) | Text with rationales | Labels + token-level explanations | Attention-based rationale extraction | Narrow domain (hate speech) |

### Existing Pipelines

| Pipeline | Method | Output |
|----------|--------|--------|
| r_Leo/ | KMeans clustering on SDG scores | Country clusters + silhouette analysis |
| r_Ting/ | TF-IDF + LinearSVC | Multi-label SDG classification |
| r_Aiden/ | BERT + attention | Hate speech classification + rationale extraction |

### Integration Opportunities

1. **SDG Text ↔ Country Data**: SDGi has `metadata.country`, GSP has country-level scores → can join on country code
2. **Classification ↔ Clustering**: r_Ting classifies SDG text, r_Leo clusters countries → could predict cluster membership from text
3. **Multi-label ↔ Explanation**: SDGi has 17 labels (same as SDGs), HateXplain has rationale extraction → could explain SDG predictions
4. **Temporal**: SDGi metadata includes `year` (2016-2023) — currently unused

---

## Proposed AI Solutions

---

### Solution 1: SDG Text Explainability

**Problem Statement**

SDGi classifies text into 17 SDGs but provides no explanation for predictions. Users cannot audit why a document is assigned to specific goals.

**Core Idea**

Transfer the rationale extraction technique from HateXplain (attention-based token importance) to SDG classification.

**Data & Components Used**

- SDGi text data (training + test)
- r_Aiden attention mechanism architecture
- SDGi 17-label schema

**Proposed Pipeline**

1. Fine-tune BERT classifier on SDGi (17 labels, multi-label)
2. Add attention layer for rationale extraction (from r_Aiden)
3. During inference: extract attention weights per token
4. Output: predicted labels + highlighted rationale tokens per label

**Expected Output**

For each input text document:
- Predicted SDG labels (1–17)
- Token-level importance scores showing which words/segments support each prediction
- Visual highlighting of rationale phrases

**Why This Is Interesting**

- First SDG classifier with built-in explainability
- Enables auditing and trust in automated SDG labeling
- Transferable technique to other text classification domains
- Addresses "black box" concern in AI for social good

**Feasibility**

High — architecture directly adapted from r_Aiden; only domain adaptation needed

---

### Solution 2: Country SDG Gap Analysis

**Problem Statement**

Countries have varying SDG performance levels, but we don't know what textual topics explain the gaps between high and low performers.

**Core Idea**

Link SDG text topic patterns (from SDGi) to country-level performance clusters (from GSP) to identify which themes differentiate country groups.

**Data & Components Used**

- GSP country SDG scores + r_Leo clusters
- SDGi text + metadata.country
- Topic modeling (LDA or embedding-based clustering)

**Proposed Pipeline**

1. Extract topics from all SDGi text (per country) using LDA or sentence embeddings
2. Aggregate topic distribution per country
3. Join with GSP country scores and cluster assignments
4. Statistical analysis: which topics correlate with high/low SDG clusters?

**Expected Output**

- Topic-by-cluster matrix showing SDG themes that differentiate country groups
- Key findings: e.g., "countries focusing on Gender (SDG 5) text tend to be in high-performance clusters"

**Why This Is Interesting**

Bridges qualitative text content with quantitative outcomes — connects what countries write about to how they perform

**Feasibility**

Medium — requires topic modeling + correlation analysis; straightforward statistical methods

---

### Solution 3: Cross-Domain Harmful Content Detection

**Problem Statement**

Hate speech detection techniques exist but are narrow. Could adapt to detect subtle harm in SDG-related content, such as greenwashing or SDG washing.

**Core Idea**

Fine-tune HateXplain model on SDG-related texts with new labels for misleading claims.

**Data & Components Used**

- HateXplain trained model (from r_Aiden)
- SDGi text subset for fine-tuning
- New annotation: "genuine" vs "SDG washing" vs "greenwashing"

**Proposed Pipeline**

1. Use r_Aiden-trained model as base
2. Select and annotate subset of SDGi texts for new task (require human annotation)
3. Fine-tune classifier with new labels
4. Add rationale extraction for flagged issues

**Expected Output**

- Classification of SDG texts as genuine vs misleading
- Rationale extraction explaining why content flagged

**Why This Is Interesting**

Novel application addressing real-world SDG credibility issues; first step toward automated SDG claim verification

**Feasibility**

Medium — depends on annotation quality; model adaptation straightforward

---

### Solution 4: Multimodal SDG Benchmarking

**Problem Statement**

Single-metric country scores don't capture complexity of SDG performance.

**Core Idea**

Combine text topics, classification results, and tabular scores for richer country benchmarking.

**Data & Components Used**

- GSP tabular scores
- SDGi text + classification results
- r_Leo clustering + r_Ting classification

**Proposed Pipeline**

1. Classify all SDGi text by country → topic/label distribution per country
2. Join with GSP scores
3. Create multi-view country profiles: text topics + score profile + cluster
4. Analyze: which topic patterns associate with high/low scores across all 17 SDGs?

**Expected Output**

Country profiles containing:
- Topic composition (from text)
- SDG score profile (from GSP)
- Cluster assignment
- Key differentiators

**Why This Is Interesting**

Multi-source benchmarking beyond simple scores; reveals relationship between reporting focus and outcomes

**Feasibility**

Medium — data join requires country code alignment; multi-view analysis is standard

---

### Solution 5: Temporal SDG Trend Analysis

**Problem Statement**

SDGi contains temporal data (2016-2023) but current pipelines ignore the time dimension.

**Core Idea**

Track how SDG focus shifts over time per country to identify emerging and declining priorities.

**Data & Components Used**

- SDGi text + metadata.year
- r_Ting classification pipeline
- Time series analysis

**Proposed Pipeline**

1. Classify all SDGi text with timestamps
2. Aggregate label distribution per country per year
3. Compute trend: for each country-SDG pair, calculate slope over time
4. Identify: which SDGs gaining/losing focus globally and per country

**Expected Output**

- Time series charts of SDG focus per country
- Global trend indicators: e.g., "SDG 13 (Climate Action) increased 40% from 2016-2023"
- Country-specific anomaly detection

**Why This Is Interesting**

First temporal analysis of SDG reporting trends; enables forecasting and priority shifting

**Feasibility**

High — reuse r_Ting classifier; add temporal aggregation (standard time series)

---

## Recommended Direction

### Primary: Solution 1 (SDG Text Explainability)

**Rationale**:
- Highest feasibility — architecture already exists in r_Aiden
- Clear novelty — no existing SDG classifier with built-in explainability
- Directly builds on two assets (SDGi + HateXplain pipelines)
- Measurable evaluation: precision/recall on rationale extraction

### Backup: Solution 2 (Country SDG Gap Analysis)

**Rationale**:
- Bridges text and numeric data meaningfully
- Strong analytical value for policy audiences
- Uses all three assets

---

## Evaluation Criteria

For each solution, evaluate by:

1. **Originality**: Does it go beyond existing pipelines?
2. **Feasibility**: Can it be built within MSc time constraints?
3. **Clarity**: Is the evaluation metric clear?
4. **Impact**: Does it address a real sustainable development problem?