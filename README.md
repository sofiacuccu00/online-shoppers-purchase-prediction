# Online Shoppers Purchasing Intention — Supervised Learning

Project 1 for the course *Machine Learning (Reykjavik University)*.

We apply supervised learning to predict whether an online shopping session
will end with a purchase, using the UCI **Online Shoppers Purchasing Intention**
dataset (id=468).

## Repository structure

- `notebooks/P1_SupervisedLearning.ipynb` — main pipeline: data loading, EDA,
  preprocessing, hyperparameter tuning, final evaluation (raw vs balanced).
- `notebooks/SMOTE-Supervised_Learning.ipynb` — additional experiment: SMOTE
  (Synthetic Minority Over-sampling Technique) as an alternative to
  `class_weight='balanced'` for handling the class imbalance. It generates
  synthetic samples of the minority class during training and is applied
  inside each cross-validation fold to avoid data leakage.

## How to run

The notebooks are designed to run on **Google Colab** without any local setup.
The dataset is fetched at runtime from the UCI Machine Learning Repository
via the `ucimlrepo` package, so no manual download is required.

1. Download the notebooks
2. Open the notebook you want to run in Google Colab
3. Select **Runtime → Run all**
4. Wait for the pipeline to complete.

All experiments use a fixed `random_state=42` for the train/test split, the
cross-validation splits and the classifiers, so re-running the notebooks
reproduces the reported results.

## Data

The dataset is downloaded directly from the UCI Machine Learning Repository
via the `ucimlrepo` package (id=468).

- 12,330 sessions
- 17 features (10 numerical, 6 categorical, 1 binary)
- Binary target: `Revenue` (15.5% positive)

## Pipeline

- Stratified 80/20 train-test split
- Outlier analysis (IQR with 2.5×IQR, but no filtering)
- One-Hot Encoding for categorical features, Min-Max scaling for numerical
- 6 classifiers: Logistic Regression, Decision Tree, Random Forest, HistGB,
  SVM (RBF), KNN
- Two class-imbalance settings: `raw` (no reweighting) and `balanced`
  (`class_weight='balanced'`)
- 5-fold stratified cross-validation with `GridSearchCV` (scoring = macro-F1)
- Final evaluation on the held-out test set (accuracy, precision, recall,
  F1 on the positive class, AUC)

## Results (test set, best model)

| Setting  | Model  | F1    | AUC   |
|----------|--------|-------|-------|
| balanced | HistGB | 0.658 | 0.920 |
| raw      | HistGB | 0.655 | 0.932 |

The trivial majority-class benchmark reaches 84.5% accuracy but zero F1.

