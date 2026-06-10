# 📊 Customer Churn Analysis — Machine Learning Dashboard
[![Open in Streamlit]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://umairnaqvi04-churn-analytics.streamlit.app)](https://churn-analytics-ml-projec-6c87hlzf8wph69pfb8p4lq.streamlit.app/)



> An interactive machine learning project that predicts customer churn using multiple classification algorithms and K-Means clustering, with a fully designed frontend dashboard.

---

## 👨‍💻 Developer

**Syed Umair**
OEL Project — Machine Learning & Data Science

---

## 📌 Project Overview

This project analyzes a bank's customer data to predict which customers are likely to leave (churn). It covers the full ML pipeline — from data loading and preprocessing to model training, evaluation, and customer segmentation.

---

## 🗂️ Project Structure

```
📦 Customer-Churn-Analysis
 ┣ 📓 Customer_Churn_Analysis_Syed_Umair.ipynb   ← Main Colab Notebook
 ┣ 📄 OEL_Churn_Dashboard.py                     ← Interactive HTML Dashboard (Colab-ready)
 ┣ 📄 README.md                                  ← Project Documentation
 ┗ 📊 Customer-Churn-Records.csv                 ← Dataset (upload manually in Colab)
```

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| File | `Customer-Churn-Records.csv` |
| Rows | 10,000 |
| Features | 14 |
| Target Column | `Exited` (1 = Churned, 0 = Retained) |
| Churn Rate | ~20.4% |

**Key Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

---

## ⚙️ ML Pipeline

1. **Data Loading** — pandas se CSV load
2. **EDA** — shape, null values, column info
3. **Feature Dropping** — RowNumber, CustomerId, Surname remove
4. **Label Encoding** — Geography, Gender ko numeric mein convert
5. **Train/Test Split** — 80% train, 20% test
6. **Feature Scaling** — StandardScaler
7. **Model Training** — 4 classifiers train kiye
8. **Evaluation** — accuracy, confusion matrix, classification report
9. **Clustering** — K-Means (k=3) se customer segments

---

## 🤖 Models Used

| Model | Accuracy |
|-------|----------|
| Decision Tree | 79.5% |
| Random Forest | **86.5% 🏆** |
| KNN | 82.3% |
| Naive Bayes | 75.1% |

> ✅ **Best Model: Random Forest** with 86.5% accuracy

---

## 👥 Customer Segments (K-Means)

| Cluster | Name | Description |
|---------|------|-------------|
| 0 | Low-Risk | Young, low balance, active users |
| 1 | High-Value Stable | Mid-age, high balance, long tenure |
| 2 | At-Risk Churners | Older, high balance, inactive |

---

## 🖥️ Interactive Dashboard

Project mein ek fully designed interactive dashboard bhi hai (`OEL_Churn_Dashboard.py`) jisme yeh tabs hain:

- **Overview** — Key metrics aur churn distribution
- **Data Analysis** — Balance, geography, correlation heatmap
- **ML Pipeline** — Step-by-step visual pipeline
- **Model Results** — All 4 models ki accuracy comparison
- **Clustering** — K-Means scatter plot
- **Source Code** — Syntax-highlighted code

### Dashboard Colab mein chalane ka tarika:

```python
# OEL_Churn_Dashboard.py ka saara code ek Colab cell mein paste karo
# Phir Shift+Enter dabao
```

---

## 🛠️ Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
IPython.display
requests
```

---

## 🚀 How to Run

1. [Google Colab](https://colab.research.google.com/) kholo
2. `Customer_Churn_Analysis_Syed_Umair.ipynb` upload karo
3. `Customer-Churn-Records.csv` dataset upload karo
4. **Runtime → Run All** click karo
5. Dashboard ke liye `OEL_Churn_Dashboard.py` ka code alag cell mein paste karo

---

## 📜 License

This project is for educational purposes — OEL submission.

---

*Developed by **Syed Umair***
