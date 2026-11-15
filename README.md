# CodeAlpha Task 1 — Credit Scoring Model

This internship project compares Logistic Regression and Random Forest on historical German credit records. The positive class is high credit risk. It is an educational model based on a small historical dataset.



## How it works

1. Load the 1,000 records and map original labels 1 and 2 to low risk (0) and high risk (1).
2. Created `amount_per_month` by dividing the credit amount by the term in months. This is a derived amount and excludes interest.
3. Split the data into 800 training and 200 test records. Stratification preserves the class proportions.
4. Fit the one-hot encoder and numeric scaler on training data using each model's pipeline. Evaluate both models on the same test records.
5. Compare precision, recall, F1, ROC-AUC and the dataset's error cost. Precision measures how many flagged cases were high risk. Recall measures how many high-risk cases were caught. ROC-AUC measures how well the scores rank high-risk cases above low-risk cases across thresholds.

Personal status/sex, age, and foreign-worker status are excluded as modeling choices. This exclusion alone does not establish fairness.

## Results

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | UCI cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.500 | 0.780 | 0.667 | 0.533 | 0.593 | 0.793 | 156 |
| Logistic Regression | 0.167 | 0.575 | 0.400 | 0.833 | 0.541 | 0.793 | 125 |
| Random Forest | 0.500 | 0.780 | 0.722 | 0.433 | 0.542 | 0.788 | 180 |
| Random Forest | 0.167 | 0.530 | 0.380 | 0.900 | 0.535 | 0.788 | 118 |

The dataset assigns a cost of 5 to a missed high-risk record and 1 to a false high-risk flag. The exact threshold of 1/6 follows from those costs if the model probabilities are calibrated. Here it is a fixed comparison threshold and calibration was not assessed. At the default threshold, Logistic Regression had the higher F1 and ROC-AUC in this split. At the lower threshold, Random Forest had the lowest observed cost and caught 54 of the 60 high-risk records while falsely flagging 88 low-risk records.

## Data source

Hofmann, H. (1994). [Statlog (German Credit Data)](https://doi.org/10.24432/C5NC77), UCI Machine Learning Repository. The original `german.data` is reproduced without changes and distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
