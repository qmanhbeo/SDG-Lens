# Creative Solutions for Real SDG Problems

Based on detailed inspection of available datasets:
- **GSP**: 168 countries, overall SDG index scores
- **SDGi**: 7350 docs (182 countries, 2016-2023), 17 labels, 3 languages, pre-computed embeddings
- **HateXplain**: Hate speech + rationales + target communities

---

## Solution 6: Cross-Lingual SDG Classification

**Problem**
71.9% of SDGi is English, 15.9% Spanish, 12.2% French. Current SDG Lens likely trains on English-only, ignoring 28% of data.

**Real Problem**
Many countries (Latin America, Francophone Africa) only submit reports in their language. English-centric models miss their SDG priorities.

**Core Idea**
Compare multilingual models (XLM-RoBERTa, mBERT) vs English-only BERT. Quantify performance gap across languages.

**Data**
- SDGi: filter by language, use all 3 languages together
- Compare: train on EN→test on EN/ES/FR

**Pipeline**
1. Load SDGi, stratify by language
2. Train XLM-RoBERTa-base on full corpus (all 3 languages)
3. Train baseline: English-only BERT
4. Evaluate: per-language F1, macro-F1 across languages

**Expected Output**
- Language-specific precision/recall for all 17 SDGs
- Quantified gap: "XLM-R outperforms English-only by X% on Spanish texts"

**Why Interesting**
First multilingual SDG classifier benchmark. Reveals language bias in UN reporting analysis.

**Feasibility**: High — standard transformer fine-tuning

---

## Solution 7: VNR vs VLR Gap Analysis

**Problem**
Countries report at national (VNR) and local (VLR) levels — do they prioritize different SDGs?

**Real Problem**
Local governments may emphasize different goals (e.g., urban housing) than national governments (e.g., economic growth). No analysis exists.

**Core Idea**
Compare SDG label distributions between VNR (4001 docs) and VLR (1879 docs) to identify systematic differences.

**Data**
- SDGi: filter by `metadata.type` = 'vnr' vs 'vlr'
- GSP: country-level SDG scores

**Pipeline**
1. Classify all docs (or use existing labels)
2. Aggregate label distribution per type (VNR vs VLR)
3. Statistical test: Chi-squared per SDG to find significant differences
4. Per-country: compare VNR vs VLR for same country (where both exist)
5. Join with GSP: do VLR-heavy countries have different scores?

**Expected Output**
- "SDG 11 (Sustainable Cities) is 2.3x more common in VLR than VNR"
- Country-level: which countries have largest VNR/VLR divergence?
- Correlation: do VLR-heavy countries score higher on specific SDGs?

**Why Interesting**
First quantitative analysis of national vs local SDG reporting priorities. Useful for UN localization efforts.

**Feasibility**: High — simple aggregation + chi-squared

---

## Solution 8: Temporal SDG Priority Tracker

**Problem**
SDGi has year metadata (2016-2023) but nobody tracks how SDG focus shifts over time.

**Real Problem**
Policy makers need to know: which SDGs are gaining attention? Which are declining? How did COVID affect priorities?

**Core Idea**
Track SDG label distribution per year (2016-2023) to identify emerging and declining priorities.

**Data**
- SDGi: `metadata.year` (2016-2023)
- ~103 docs in 2016 → ~794 in 2023

**Pipeline**
1. Classify all docs by year (or use existing labels)
2. Aggregate: count of each SDG per year
3. Compute YoY change: percentage shift
4. Global trend: slope of each SDG over time
5. Country-level: track individual country's SDG emphasis changes

**Expected Output**
- "SDG 13 (Climate Action) increased 40% from 2016-2023"
- "SDG 16 (Peace) peaked in 2020, declined thereafter"
- Country anomalies: which countries deviate from global trend?

**Why Interesting**
First temporal analysis of UN SDG reporting. Enables forecasting and policy tracking.

**Feasibility**: High — time series aggregation

---

## Solution 9: Country Embedding Similarity

**Problem**
GSP scores countries on outcomes, but not on content similarity — which countries write about similar topics?

**Real Problem**
Policy makers want to find peer countries for learning. But similarity by SDG score doesn't capture content.

**Core Idea**
Use SDGi's pre-computed 1536-dim embeddings to cluster countries by content similarity.

**Data**
- SDGi: `embedding` column (text-embedding-ada-002)
- SDGi: `metadata.country` (182 countries)

**Pipeline**
1. Aggregate embeddings per country (mean pooling)
2. Compute cosine similarity matrix between all 182 countries
3. Clustering: hierarchical clustering or t-SNE visualization
4. Identify "content peers": countries similar by text, not just score
5. Compare with GSP cluster membership (from r_Leo)

**Expected Output**
- Country similarity matrix (182x182)
- Top-5 content-similar countries for each country
- Comparison: "Countries A and B have similar SDG scores but different content"
- Visualization: t-SNE of countries by embedding

**Why Interesting**
New way to find policy peers beyond score-based rankings. Enables cross-country learning.

**Feasibility**: High — embedding aggregation + similarity

---

## Solution 10: Zero-Shot SDG Classifier

**Problem**
Labeling new SDG documents is expensive. Can LLMs replace supervised training?

**Real Problem**
Organizations want quick SDG classification without training. Need to test if LLMs can do it zero-shot.

**Core Idea**
Use LLM (GPT-4o mini via API or local LLM) as zero-shot classifier, compare with supervised BERT.

**Data**
- SDGi: sample 200 docs for evaluation
- Prompt engineering: provide SDG definitions

**Pipeline**
1. Create prompt: "Classify this text into SDGs 1-17. Here are definitions..."
2. Query LLM for each doc (batch for efficiency)
3. Compare with SDGi ground truth labels
4. Compute F1, per-SDG precision/recall
5. Error analysis: which SDGs does LLM struggle with?

**Expected Output**
- "GPT-4o achieves X% F1 vs BERT's Y% F1"
- Per-SDG breakdown: which goals are hard for zero-shot?
- Cost analysis: $ per document

**Why Interesting**
Tests if LLMs can replace human labeling. Practical value for organizations.

**Feasibility**: Medium — requires API access or local LLM

---

## Solution 11: SDG Claim Verification (Greenwashing Detection)

**Problem**
Corporations and governments may overstate SDG commitments — "SDG washing".

**Real Problem**
No automated way to detect misleading SDG claims. HateXplain has harmful content detection — adapt it.

**Core Idea**
Fine-tune HateXplain model on SDG-related text with new labels: "genuine" vs "SDG washing".

**Data**
- SDGi: use subset for fine-tuning (need annotation)
- Annotate: mark texts as genuine vs misleading
- Requires human annotation (graduate students or crowd workers)

**Pipeline**
1. Select 500 SDGi docs randomly
2. Annotate: label as "genuine" or "SDG washing" with rationale
3. Fine-tune HateXplain model (BERT + attention) on new task
4. Evaluate: precision/recall on claim verification
5. Extract rationales: which tokens indicate greenwashing?

**Expected Output**
- Binary classifier: genuine vs SDG washing
- Per-class F1 scores
- Rationale extraction: "Company X uses vague language like 'committed to sustainability'"

**Why Interesting**
First automated greenwashing detection for SDGs. High real-world value for watchdog organizations.

**Feasibility**: Medium — requires annotation effort

---

## Priority Ranking

| Solution | Problem Realness | Feasibility | Impact | Priority |
|----------|-----------------|-------------|--------|----------|
| 7. VNR vs VLR Gap | High | High | High | 1 |
| 8. Temporal Tracker | High | High | High | 1 |
| 9. Country Embedding | Medium | High | Medium | 2 |
| 6. Cross-Lingual | High | High | High | 2 |
| 10. Zero-Shot | Medium | Medium | Medium | 3 |
| 11. Greenwashing | High | Medium | High | 3 |

**Recommended**: Start with Solution 7 or 8 — highest real problem + straightforward implementation.