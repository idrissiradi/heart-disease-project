# Heart Disease Classification — End-to-End ML Project
 
Binary classification project predicting the presence of heart disease from clinical data. Compares three classifiers, applies hyperparameter tuning with `RandomizedSearchCV` and `GridSearchCV`, and evaluates final model performance with cross-validation and feature importance analysis.
 
---
 
## Results
 
### Model Comparison (Test Set Accuracy)
 
| Model | Test Accuracy |
|---|---|
| Logistic Regression | **88.52%** ✓ best |
| Random Forest | 86.89% |
| K-Nearest Neighbors | 75.41% |
 
### Best Model — Logistic Regression (5-Fold Cross-Validation)
 
| Metric | Score |
|---|---|
| Precision | 82.08% |
| Recall | 92.12% |
| F1 Score | 86.73% |
  
---
 
## Pipeline
 
```
heart-disease.csv
      │
      ▼
EDA & Visualization
  · Class distribution
  · Feature correlations (heatmap)
  · Clinical feature analysis
      │
      ▼
Preprocessing
  · Train/test split (stratified)
  · Feature scaling
      │
      ▼
Model Comparison
  · LogisticRegression
  · KNeighborsClassifier  (best k search)
  · RandomForestClassifier
      │
      ▼
Hyperparameter Tuning
  · RandomizedSearchCV  (broad search)
  · GridSearchCV        (fine-tuned search)
      │
      ▼
Evaluation
  · Confusion matrix
  · Classification report (precision / recall / F1)
  · ROC Curve
  · 5-fold cross-validation
  · Feature importance (Random Forest)
```
 
---
 
## Dataset
 
[Cleveland Heart Disease — UCI / Kaggle](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
 
- 303 patient records, 14 features
- Target: `target` — 0 (no disease), 1 (disease present)
- Features: age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar, ECG results, max heart rate, exercise-induced angina, ST depression, slope, vessels, thalassemia
 
---
 
## Stack
 
- **Python 3.x**, Jupyter Notebook
- **scikit-learn** — LogisticRegression, KNeighborsClassifier, RandomForestClassifier, RandomizedSearchCV, GridSearchCV, cross_val_score
- **pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — EDA visualization, ROC curve, confusion matrix, feature importance
 
---
 
## Setup
 
```bash
git clone https://github.com/idrissiradi/heart-disease-project
cd heart-disease-project
pip install -r requirements.txt
jupyter notebook end-to-end-heart-disease-classification.ipynb
```
 
---

## Project Structure

```
heart-disease-project/
├── end-to-end-heart-disease-classification.ipynb  # Main analysis notebook
├── heart-disease.csv                               # Dataset
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

---

This is a learning project for educational purposes.
