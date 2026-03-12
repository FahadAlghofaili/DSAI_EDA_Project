"""
Saudi Arabia Land Price — Streamlit Dashboard
==============================================
Interactive dashboard for exploring Saudi land price data,
visualizing EDA findings, and predicting land prices using
a trained Ridge Regression model.

Author: Fahad Alghofaili
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import arabic_reshaper
from bidi.algorithm import get_display

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Saudi Land Price Analysis",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and cache the cleaned dataset."""
    try:
        df = pd.read_csv("dataset_clean.csv")
        return df
    except FileNotFoundError:
        st.error("❌ dataset_clean.csv not found. Please ensure the file is in the project directory.")
        return None

# دالة معالجة النصوص العربية للرسوم البيانية
def fix_arabic(text):
    if isinstance(text, str):
        reshaped_text = arabic_reshaper.reshape(text)
        return get_display(reshaped_text)
    return text

@st.cache_resource
def load_model():
    """Load and cache the trained Ridge Regression model."""
    try:
        model = joblib.load("land_price_model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


def format_currency(value):
    """Format a number as SAR currency."""
    if abs(value) >= 1_000_000:
        return f"{value:,.0f} SAR ({value / 1_000_000:,.2f}M)"
    return f"{value:,.0f} SAR"


def build_feature_vector(model, mainlocation, neighborhood, frontage, purpose,
                         streetwidth, size):
    """
    Build a one-hot encoded feature vector matching the model's expected features.
    NOTE: streetwidth and size are log-transformed to match training data preprocessing.
    """
    feature_names = list(model.feature_names_in_)
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)

    # Set numeric features — apply log transform to match training preprocessing
    if "streetwidth" in feature_names:
        input_data["streetwidth"] = np.log(streetwidth)
    if "size" in feature_names:
        input_data["size"] = np.log(size)

    # Set one-hot encoded categorical features
    location_col = f"mainlocation_{mainlocation}"
    if location_col in feature_names:
        input_data[location_col] = 1

    neighborhood_col = f"neighborhood_{neighborhood}"
    if neighborhood_col in feature_names:
        input_data[neighborhood_col] = 1

    frontage_col = f"frontage_{frontage}"
    if frontage_col in feature_names:
        input_data[frontage_col] = 1

    purpose_col = f"purpose_{purpose}"
    if purpose_col in feature_names:
        input_data[purpose_col] = 1

    return input_data

# ──────────────────────────────────────────────
# Custom CSS Styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 {
        font-size: 0.85rem;
        margin: 0;
        opacity: 0.9;
    }
    .metric-card h2 {
        font-size: 1.5rem;
        margin: 0.3rem 0 0 0;
        font-weight: 700;
    }
    /* Section divider */
    .section-divider {
        border-top: 2px solid #E8ECF1;
        margin: 2rem 0;
    }
    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .prediction-box h2 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .prediction-box p {
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
st.sidebar.markdown("## 🏗️ Land Price EDA")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigate",
    [
        "📈 EDA Visualizations",
        "📉 Insights & Statistics",
        "🤖 Model Prediction",
    ],
)

# Load data
df = load_data()

# ──────────────────────────────────────────────
# PAGE 3: EDA Visualizations
# ──────────────────────────────────────────────
if page == "📈 EDA Visualizations":
    st.markdown('<p class="main-header">📈 EDA Visualizations</p>', unsafe_allow_html=True)
    st.markdown("---")

    if df is not None:
        # Use a reasonable subset — cap extreme outliers for better visualization
        viz_df = df[df["land_price"] <= df["land_price"].quantile(0.95)].copy()

        # ---- Chart 1: Land Price Distribution ----
        st.markdown("### 1️⃣ Land Price Distribution")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.histplot(viz_df["land_price"], bins=50, kde=True, color="#667eea", ax=ax1)
        ax1.set_xlabel("Land Price (SAR)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.set_title("Distribution of Land Prices (Top 95%)", fontsize=14, fontweight="bold")
        ax1.ticklabel_format(style="plain", axis="x")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        st.markdown("---")

        # ---- Chart 2: Top Cities by Average Price ----
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### 2️⃣ Top 10 Cities by Avg Land Price")
            city_avg = df.groupby("mainlocation")["land_price"].mean().sort_values(ascending=False).head(10)
            
            # معالجة أسماء المدن بالعربي
            city_indices_ar = [fix_arabic(city) for city in city_avg.index[::-1]]
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            bars = ax2.barh(city_indices_ar, city_avg.values[::-1], color=sns.color_palette("viridis", 10))
            ax2.set_xlabel("Average Land Price (SAR)", fontsize=11)
            ax2.set_title("Top 10 Cities by Avg Land Price", fontsize=13, fontweight="bold")
            ax2.ticklabel_format(style="plain", axis="x")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        with col_b:
            st.markdown("### 3️⃣ Purpose Breakdown")
            purpose_counts = df["purpose"].value_counts()
            
            # معالجة أسماء "الغرض" بالعربي
            purpose_labels_ar = [fix_arabic(p) for p in purpose_counts.index]
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            colors = ["#667eea", "#764ba2", "#f093fb", "#c3cfe2"]
            wedges, texts, autotexts = ax3.pie(
                purpose_counts.values,
                labels=purpose_labels_ar,
                autopct="%1.1f%%",
                colors=colors,
                startangle=140,
                textprops={"fontsize": 10},
            )
            ax3.set_title("Land Purpose Distribution", fontsize=13, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)

        st.markdown("---")



        # ---- Chart 4: Street Width Impact on Price ----
        st.markdown("### 4️⃣ Street Width Impact on Price")
        fig6, ax6 = plt.subplots(figsize=(10, 5))
        # Bin street width into groups
        bins = [0, 15, 20, 30, 40, 60, 101]
        labels = ["≤15m", "16-20m", "21-30m", "31-40m", "41-60m", "61-100m"]
        viz_df2 = df[df["land_price"] <= df["land_price"].quantile(0.95)].copy()
        viz_df2["sw_group"] = pd.cut(viz_df2["streetwidth"], bins=bins, labels=labels, right=True)
        group_avg = viz_df2.groupby("sw_group", observed=False)["land_price"].mean()
        ax6.bar(group_avg.index, group_avg.values, color=sns.color_palette("mako", len(group_avg)))
        ax6.set_xlabel("Street Width Group", fontsize=12)
        ax6.set_ylabel("Average Land Price (SAR)", fontsize=12)
        ax6.set_title("Avg Land Price by Street Width", fontsize=14, fontweight="bold")
        ax6.ticklabel_format(style="plain", axis="y")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

        st.markdown("---")

        # ---- Chart 5: Frontage Direction ----
        st.markdown("### 5️⃣ Average Total Price by Frontage Type")
        frontage_avg = df.groupby("frontage")["land_price"].mean().sort_values(ascending=False)
        
        # معالجة الواجهة بالعربي
        frontage_indices_ar = [fix_arabic(f) for f in frontage_avg.index]
        
        fig7, ax7 = plt.subplots(figsize=(10, 4))
        ax7.bar(frontage_indices_ar, frontage_avg.values, color=sns.color_palette("rocket", len(frontage_avg)))
        ax7.set_xlabel("Frontage Type", fontsize=12)
        ax7.set_ylabel("Average Total Land Price (SAR)", fontsize=12)
        ax7.set_title("Average Total Price by Frontage Type", fontsize=14, fontweight="bold")
        ax7.ticklabel_format(style="plain", axis="y")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)


# ──────────────────────────────────────────────
# باقي كود الصفحات (Insights & Model Prediction) يستمر كما هو...
# ──────────────────────────────────────────────
elif page == "📉 Insights & Statistics":
    st.markdown('<p class="main-header">📉 Insights & Statistics</p>', unsafe_allow_html=True)
    st.markdown("---")

    if df is not None:
        # Key metrics
        st.markdown("### 📊 Key Metrics")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Listings</h3>
                <h2>{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Price/m²</h3>
                <h2>{df['Pricepm'].mean():,.0f} SAR</h2>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Median Land Price</h3>
                <h2>{df['land_price'].median():,.0f} SAR</h2>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Cities Covered</h3>
                <h2>{df['mainlocation'].nunique()}</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Descriptive statistics
        st.markdown("### 📋 Descriptive Statistics")
        numeric_stats = df.describe().T
        numeric_stats.columns = ["Count", "Mean", "Std Dev", "Min", "25%", "Median", "75%", "Max"]
        st.dataframe(numeric_stats.style.format("{:,.2f}"), use_container_width=True)

        st.markdown("---")

        # Key findings
        st.markdown("### 💡 Key Findings")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            #### 🏙️ Location Insights
            - ** (Riyadh)** dominates with **71%** of all listings
            - Top 3 cities: Riyadh, Al Khobar, Jeddah
            - Al Diriyah and Al Madinah Al Munawwarah round out the top 5

            #### 💰 Price Insights
            - Median land price: **~1.87M SAR**
            - Average price per m²: **~2,626 SAR**
            - Large price variance indicates diverse market segments
            """)

        with col2:
            st.markdown("""
            #### 🏠 Purpose Insights
            - **Residential (سكني)**: ~72% of listings
            - **Mixed-use (سكني أو تجاري)**: ~16%
            - **Commercial (تجاري)**: ~12%
            - Commercial land has higher avg price/m²

            #### 🛣️ Street Width Insights
            - Multi-street frontage (3-4 streets) commands premium
            - Average street width: ~24.7m
            """)

        st.markdown("---")

        # Per-city summary
        st.markdown("### 🏙️ Top 10 Cities Summary")
        city_summary = (
            df.groupby("mainlocation")
            .agg(
                Listings=("land_price", "count"),
                Avg_Price_per_m2=("Pricepm", "mean"),
                Median_Total_Price=("land_price", "median"),
                Avg_Size_m2=("size", "mean"),
            )
            .sort_values("Listings", ascending=False)
            .head(10)
        )
        city_summary.columns = ["Listings", "Avg Price/m² (SAR)", "Median Price (SAR)", "Avg Size (m²)"]
        st.dataframe(
            city_summary.style.format({
                "Listings": "{:,}",
                "Avg Price/m² (SAR)": "{:,.0f}",
                "Median Price (SAR)": "{:,.0f}",
                "Avg Size (m²)": "{:,.0f}",
            }),
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# PAGE 5: Model Prediction
# ──────────────────────────────────────────────
elif page == "🤖 Model Prediction":
    st.markdown('<p class="main-header">🤖 Land Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Use the trained Ridge Regression model to predict land prices</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    model = load_model()

    if df is not None and model is not None:
        st.markdown("### 📝 Enter Land Details")

        # Row 1: Categorical inputs
        cat_col1, cat_col2 = st.columns(2)

        with cat_col1:
            # City selection
            cities = sorted(df["mainlocation"].unique())
            selected_city = st.selectbox("🏙️ City (mainlocation)", cities, index=cities.index("الرياض") if "الرياض" in cities else 0)

            # Dynamic neighborhood filtering based on selected city
            city_neighborhoods = sorted(df[df["mainlocation"] == selected_city]["neighborhood"].unique())
            selected_neighborhood = st.selectbox("🏘️ Neighborhood", city_neighborhoods)

        with cat_col2:
            # Frontage selection
            frontages = sorted(df["frontage"].unique())
            selected_frontage = st.selectbox("🧭 Frontage Direction", frontages)

            # Purpose selection
            purposes = sorted(df["purpose"].unique())
            selected_purpose = st.selectbox("📋 Purpose", purposes)

        st.markdown("---")

        # Row 2: Numeric inputs
        st.markdown("### 📐 Enter Numeric Features")
        num_col1, num_col2 = st.columns(2)

        with num_col1:
            street_width = st.number_input(
                "🛣️ Street Width (m)",
                min_value=1.0,
                max_value=200.0,
                value=20.0,
                step=1.0,
                help="Width of the adjacent street in meters",
            )

        with num_col2:
            land_size = st.number_input(
                "📏 Land Size (m²)",
                min_value=50,
                max_value=10_000_000,
                value=750,
                step=50,
                help="Total land area in square meters",
            )

        st.markdown("---")

        # Prediction button
        if st.button("🔮 Predict Land Price", use_container_width=True, type="primary"):
            try:
                # Build the feature vector
                input_data = build_feature_vector(
                    model,
                    selected_city,
                    selected_neighborhood,
                    selected_frontage,
                    selected_purpose,
                    street_width,
                    land_size,
                )

                # Make prediction
                predicted_log_price = model.predict(input_data)[0]
                predicted_price = np.exp(predicted_log_price)

                # Ensure non-negative prediction
                predicted_price = max(0, predicted_price)

                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                    <p>Predicted Land Price</p>
                    <h2>💰 {predicted_price:,.0f} SAR</h2>
                    <p>{predicted_price / 1_000_000:,.2f} Million SAR</p>
                </div>
                """, unsafe_allow_html=True)

                # Show input summary
                st.markdown("---")
                st.markdown("### 📋 Input Summary")
                summary_cols = st.columns(2)
                with summary_cols[0]:
                    st.markdown(f"""
                    | Feature | Value |
                    |---------|-------|
                    | **City** | {selected_city} |
                    | **Neighborhood** | {selected_neighborhood} |
                    | **Frontage** | {selected_frontage} |
                    | **Purpose** | {selected_purpose} |
                    """)
                with summary_cols[1]:
                    st.markdown(f"""
                    | Feature | Value |
                    |---------|-------|
                    | **Street Width** | {street_width} m |
                    | **Land Size** | {land_size:,} m² |
                    | **Predicted Price** | {predicted_price:,.0f} SAR |
                    """)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.info("Please check that all input values are valid and try again.")



# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Saudi Land Price EDA © 2025</small>",
    unsafe_allow_html=True,
)
