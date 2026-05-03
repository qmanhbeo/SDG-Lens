# SDGi replication

## 1. Introduction 
**supervised baseline**：

- **TF-IDF and linear SVM-style multi-class classifier**
- code：`sdgi_replication.py`

This is a **partial replication**, not exact reproduction.

## 2. Getting Start
### a) Download the zip on Github

### b) Set up the enviroment
Firstly, create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Secondly, install dependencies
```bash
pip install -r requirements_sdgi.txt
```
Thirdly, run with a single command
```bash
python sdgi_replication.py
```


## 3. What does the script do automatically?
The script automatically performs the following steps:

1. Load the `UNDP/sdgi-corpus`
2. Use the predefined `train/test split`
3. Filter by `size` and `language`
4. Perform lightweight text pre-processing
5. Construct TF-IDF representations
6. Train a multi-label linear classifier
7. Evaluate on the test set
8. Save the results file and images

## 4. output result
An output folder will be created after execution（`sdgi_outputs/`）, including：

- `summary.json`
- `train_report.txt`
- `test_report.txt`
- `test_predictions.csv`
- `tfidf_vectorizer.joblib`
- `robust_multilabel_sgd.joblib`
- `figure_test_metrics.png`
- `figure_macro_f1_vs_paper.png`
- `figure_label_distribution.png`

