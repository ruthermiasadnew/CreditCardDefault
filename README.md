# 🧠 Credit Card Default Risk Prediction

This project builds and evaluates machine learning models to predict the likelihood of credit card default based on customer transaction and account history data.

---

## 📌 Overview

The goal of this project is to identify high-risk credit card customers using financial, historical, and behavioral data. It includes:
- Comprehensive data cleaning & transformation
- Feature engineering and missing value imputation
- Model building with Logistic Regression, Random Forest, and XGBoost
- Model evaluation using **AUROC** as the primary scoring metric

---

## 🧪 Workflow

### 1. **Data Cleaning**
- Date differences calculated from transaction timestamps
- Binarization of categorical indicators (e.g. account activity/status)
- Mean/mode imputation for missing data
- Encoding of string values into numerical features
- Automated filling of structured and semi-structured fields

👉 Full cleaning pipeline available in `data_cleaning.ipynb`

### 2. **Modeling Pipeline**
- `ColumnTransformer` for Yeo-Johnson power transforms
- Feature selection via `SelectFromModel` using Random Forest
- GridSearchCV used to tune:
  - Logistic Regression (`penalty`, `C`, `class_weight`)
  - Random Forest (`n_estimators`, `max_depth`)
  - XGBoost (`learning_rate`, `n_estimators`, `max_depth`)
- Evaluation with 5-fold cross-validation (AUROC scoring)

---

## 📈 Results

| Model              | Best AUROC Score |
|-------------------|------------------|
| XGBoostClassifier | `0.87` (example) |
| RandomForestClassifier | `0.85` |
| LogisticRegression | `0.81` |

✅ **XGBoost** consistently ranked highest based on AUROC in cross-validation.

---

## 🧰 Tools & Libraries
- Python (Pandas, NumPy)
- Scikit-learn (Pipeline, GridSearchCV, Feature Selection)
- XGBoost
- Jupyter Notebook

---

## 📂 Files
- `credit_default_model.ipynb`: Full model pipeline, training & ranking
- `data_cleaning.ipynb`: Detailed data processing and feature engineering
- `Portfolio Data.csv`: Sample data (not uploaded if confidential)

---

## 🔮 Future Improvements
- SHAP or LIME explainability for model interpretation
- Deployment-ready streamlit app for interactive risk prediction
- Evaluate using other metrics like F1-score, recall (for imbalanced settings)

---

## 💬 Contact
Created by **Ruth Adnew** | Bioinformatics & Data Science  
📍 Augustana University | [GitHub](https://github.com/yourusername)
