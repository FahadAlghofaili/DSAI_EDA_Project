# 🏗️ Saudi Arabia Land Price — Exploratory Data Analysis (EDA)

A comprehensive **Exploratory Data Analysis** project on Saudi Arabian land prices, featuring interactive visualizations, statistical insights, and a machine learning model for price prediction — all presented through a professional **Streamlit dashboard**.

---

## 📋 Table of Contents

- [Project Description](#-project-description)
- [Dataset Overview](#-dataset-overview)
- [EDA Explanation](#-eda-explanation)
- [Key Insights](#-key-insights)
- [Technologies Used](#-technologies-used)
- [How to Run the Project](#-how-to-run-the-project)
- [How to Launch the Dashboard](#-how-to-launch-the-dashboard)
- [Model Prediction](#-model-prediction)

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

## 🔍 EDA Explanation

The Exploratory Data Analysis follows a structured approach:

1. **Data Loading & Cleaning** — Loading the CSV dataset, handling missing values, and ensuring correct data types.
2. **Statistical Summary** — Computing descriptive statistics (mean, median, standard deviation) for numeric features.
3. **Univariate Analysis** — Exploring individual features through histograms and frequency distributions.
4. **Bivariate Analysis** — Investigating relationships between features (e.g., size vs. price, street width vs. price).
5. **Categorical Analysis** — Breaking down prices by city, purpose, and frontage direction.
6. **Correlation Analysis** — Building a correlation heatmap to identify the most influential features.
7. **Feature Engineering** — Creating one-hot encoded features for model training.

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

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.x** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Static data visualizations |
| **Seaborn** | Statistical visualizations |
| **Scikit-learn** | Machine learning (Ridge Regression) |
| **Streamlit** | Interactive web dashboard |
| **Joblib** | Model serialization/loading |

---

## 🚀 How to Run the Project

### Prerequisites

Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/FahadAlghofaili/DSAI_EDA_Project.git
cd DSAI_EDA_Project
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn streamlit joblib
```

### 3. Run the Analysis Notebook

Open and run the Jupyter notebook for the full EDA:

```bash
jupyter notebook Fahad_Alghofaili_unit_3_summary_draft.ipynb
```

---

## 🖥️ How to Launch the Dashboard

Launch the interactive Streamlit dashboard with:

```bash
streamlit run App.py
```

The dashboard will open in your browser (default: `http://localhost:8501`) and includes:

- 📋 **Project Overview** — Project introduction and objectives
- 📊 **Dataset Preview** — Interactive table with filtering
- 📈 **EDA Visualizations** — Charts and plots from the analysis
- 📉 **Insights & Statistics** — Key metrics and summary statistics
- 🤖 **Model Prediction** — Predict land prices with your own inputs

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
3. Numeric features (`streetwidth`, `size`, `Pricepm`) are included directly.
4. In the Streamlit dashboard, users can:
   - Select a **city**, **neighborhood**, **frontage direction**, and **purpose**
   - Enter **street width**, **land size**, and **price per m²**
   - Click **Predict** to get the estimated total land price

---

## 📝 License

This project was developed as part of the **DSAI** program.

## 👤 Author

**Fahad Alghofaili**
