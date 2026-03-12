# 🏗️ Saudi Arabia Land Price — Exploratory Data Analysis (EDA)

A comprehensive **Exploratory Data Analysis** project on Saudi Arabian land prices, featuring interactive visualizations, statistical insights, and a machine learning model for price prediction — all presented through a professional **Streamlit dashboard**.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Dataset Overview](#-dataset-overview)
- [Key Insights](#-key-insights)
- [Model Prediction](#-model-prediction)
- [Deploy](#-Deploy)

---

## 📖 Project Description

This project performs a full **Exploratory Data Analysis (EDA)** on a dataset of land listings across Saudi Arabia. The analysis covers data cleaning, feature engineering, statistical summaries, and visual exploration to uncover patterns and trends in land pricing.

A **Ridge Regression** model is trained to predict land prices based on location, purpose, size, and other features. The model and all insights are presented through an interactive **Streamlit web dashboard**.

---

## 📊 Dataset Overview

| Attribute | Detail |
|-----------|--------|
| **File** | `dataset_clean.csv` |
| **Rows** | 2,948 land listings |
| **Columns** | 8 features |

### Column Descriptions

| Column | Description | Type |
|--------|-------------|------|
| `mainlocation` | City name (e.g., الرياض, جدة, الدمام) | Categorical |
| `neighborhood` | Neighborhood within the city | Categorical |
| `frontage` | Land frontage direction (e.g., شمال, جنوب, شرق, غرب) | Categorical |
| `purpose` | Land purpose (سكني, تجاري, سكني أو تجاري) | Categorical |
| `streetwidth` | Width of the adjacent street (meters) | Numeric |
| `size` | Land area in square meters (m²) | Numeric |
| `Pricepm` | Price per square meter (SAR/m²) | Numeric |
| `land_price` | Total land price in SAR | Numeric (Target) |

---

## 💡 Key Insights

- **الرياض (Riyadh)** dominates the dataset with **71%** of all listings (2,099 out of 2,948).
- **Residential (سكني)** land is the most common purpose, accounting for ~72% of listings.
- **Median land price** is approximately **1.87M SAR**, with significant outliers at the high end.
- **Commercial (تجاري)** properties have higher average price per m² than residential.
- **Street width** positively correlates with land price — wider streets command premium prices.
- **Multi-street frontage** (3/4 streets) properties are significantly more expensive.
- Land **size** is the strongest predictor of total price (expected, as price = size × price/m²).
---

## 🤖 Model Prediction

### Model Details

| Attribute | Detail |
|-----------|--------|
| **Algorithm** | Ridge Regression (L2 regularization) |
| **Features** | 323 (one-hot encoded from categorical + numeric) |
| **Target** | `land_price` (total land price in SAR) |
| **Saved As** | `land_price_model.pkl` |

### How It Works

1. The model was trained on the cleaned dataset using **Ridge Regression**.
2. Categorical features (`mainlocation`, `neighborhood`, `frontage`, `purpose`) are **one-hot encoded** into 323 binary features.
3. Numeric features (`streetwidth`, `size`) are included with log normalized.
4. In the Streamlit dashboard, users can:
   - Select a **city**, **neighborhood**, **frontage direction**, and **purpose**
   - Enter **street width**, and **land size**
   - Click **Predict** to get the estimated total land price

---

## Deploy:
https://dsai-eda-project.streamlit.app/
