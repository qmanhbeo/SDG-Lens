# Reproducing HateXplain for Explainable Hate Speech Detection


## 1. Project aim

The goal of this repository is to reproduce the main ideas of HateXplain in a way that is:

- runnable end-to-end on a single machine
- easy to inspect and evaluate
- restricted to the code required for reproduction only

The reproduction keeps the original task setting of **3-class classification**:

- hate
- offensive
- normal

It also keeps the original study’s emphasis on:

- official split logic
- BERT-based modelling
- rationale-related evaluation
- reproducibility and documentation

## 2. Original paper

**Paper**  
Mathew, B., Saha, P., Yimam, S.M., Biemann, C., Goyal, P. and Mukherjee, A. (2021) *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17), pp. 14867–14875.

## 3. Data source

This reproduction uses the public HateXplain dataset and the official post-level split information released by the original project.

**Public code/data source**  
HateXplain official repository:  
`https://github.com/hate-alert/HateXplain`

This project uses:

- the public HateXplain dataset
- the official `post_id` split logic
- deterministic split enforcement in the local pipeline

The repository does **not** upload unnecessary raw data copies. Data should be obtained from the public source above.

## 4. Repository structure

```text
hatexplain-reproduction/
├─ README.md
├─ requirements.lock.txt
├─ configs/
│  ├─ fast.yaml
│  └─ full.yaml
├─ src/
│  ├─ run_once.py
│  ├─ data.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ outputs/
│  └─ .gitkeep
├─ model_card.md
├─ .gitignore
└─ LICENSE