import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repo root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_preprocessing import preprocess_data, FEATURE_COLUMNS
from src.predict import load_model, predict_probability

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction & Analytics",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction & Analytics Dashboard")
st.markdown(
    "Production-grade customer retention intelligence engine with automated exploratory data analysis "
    "and real-time ML-powered churn risk scoring."
)

# =====================================================
# PATHS
# =====================================================

IMAGES_FOLDER = ROOT_DIR / "images"
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)
SAMPLE_DATA_PATH = ROOT_DIR / "data" / "sample_churn.csv"
MODEL_PATH = ROOT_DIR / "models" / "churn_model.joblib"

# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.header("📁 Data Source Selection")
data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["Use Demo Telco Dataset", "Upload Custom CSV"]
)

df = None

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Customer Churn CSV File",
        type=["csv"],
        help="Upload standard Telco Churn CSV containing tenure, MonthlyCharges, Contract, etc."
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
else:
    if SAMPLE_DATA_PATH.exists():
        st.sidebar.info("Using bundled 120-customer Telco evaluation dataset.")
        df = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        st.sidebar.error("Sample dataset not found.")

# =====================================================
# MAIN DASHBOARD TABS
# =====================================================

tab_eda, tab_charts, tab_predict = st.tabs([
    "📊 Dataset & Distribution",
    "📈 Visual Analytics Gallery",
    "🤖 Real-Time Churn Predictor"
])

if df is not None:
    # Clean numeric columns
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    with tab_eda:
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Attributes", df.shape[1])

        churn_rate = "N/A"
        if "Churn" in df.columns:
            yes_count = (df["Churn"].astype(str).str.strip().str.lower() == "yes").sum()
            churn_rate = f"{(yes_count / len(df)) * 100:.1f}%"
        col3.metric("Baseline Churn Rate", churn_rate)

        avg_tenure = f"{df['tenure'].mean():.1f} mo" if "tenure" in df.columns else "N/A"
        col4.metric("Average Tenure", avg_tenure)

        st.markdown("### Raw Data Sample (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)

        st.download_button(
            label="📥 Download Current Dataset (CSV)",
            data=df.to_csv(index=False),
            file_name="churn_dataset.csv",
            mime="text/csv"
        )

    with tab_charts:
        st.subheader("Automated Visualizations")

        # Layout 2 columns
        c1, c2 = st.columns(2)

        # CHART 1: CHURN DISTRIBUTION
        if "Churn" in df.columns:
            with c1:
                st.markdown("#### 1. Churn Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, x="Churn", ax=ax, palette=["#3da87a", "#e05555"])
                ax.set_title("Customer Churn Distribution")
                fig.savefig(IMAGES_FOLDER / "churn_distribution.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 2: CONTRACT VS CHURN
        if "Contract" in df.columns and "Churn" in df.columns:
            with c2:
                st.markdown("#### 2. Contract Type vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, x="Contract", hue="Churn", ax=ax, palette=["#3da87a", "#e05555"])
                ax.set_title("Contract Type vs Churn")
                fig.savefig(IMAGES_FOLDER / "contract_vs_churn.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 3: INTERNET SERVICE VS CHURN
        if "InternetService" in df.columns and "Churn" in df.columns:
            with c1:
                st.markdown("#### 3. Internet Service vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax, palette=["#3da87a", "#e05555"])
                ax.set_title("Internet Service vs Churn")
                fig.savefig(IMAGES_FOLDER / "internet_vs_churn.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 4: PAYMENT METHOD VS CHURN
        if "PaymentMethod" in df.columns and "Churn" in df.columns:
            with c2:
                st.markdown("#### 4. Payment Method vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax, palette=["#3da87a", "#e05555"])
                plt.xticks(rotation=25)
                ax.set_title("Payment Method vs Churn")
                fig.savefig(IMAGES_FOLDER / "payment_vs_churn.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 5: TENURE DISTRIBUTION
        if "tenure" in df.columns:
            with c1:
                st.markdown("#### 5. Tenure Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(df["tenure"], bins=25, kde=True, ax=ax, color="#1e3050")
                ax.set_title("Customer Tenure Distribution")
                fig.savefig(IMAGES_FOLDER / "tenure_distribution.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 6: MONTHLY CHARGES VS CHURN
        if "MonthlyCharges" in df.columns and "Churn" in df.columns:
            with c2:
                st.markdown("#### 6. Monthly Charges vs Churn")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax, palette=["#3da87a", "#e05555"])
                ax.set_title("Monthly Charges vs Churn")
                fig.savefig(IMAGES_FOLDER / "monthlycharges_vs_churn.png", bbox_inches="tight")
                st.pyplot(fig)
                plt.close(fig)

        # CHART 7: TOTAL CHARGES VS CHURN
        if "TotalCharges" in df.columns and "Churn" in df.columns:
            st.markdown("#### 7. Total Charges vs Churn")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x="Churn", y="TotalCharges", ax=ax, palette=["#3da87a", "#e05555"])
            ax.set_title("Total Charges vs Churn")
            fig.savefig(IMAGES_FOLDER / "totalcharges_vs_churn.png", bbox_inches="tight")
            st.pyplot(fig)
            plt.close(fig)

else:
    with tab_eda:
        st.info("Please select a data source or upload a CSV file in the sidebar to begin.")

# =====================================================
# REAL-TIME PREDICTOR TAB
# =====================================================

with tab_predict:
    st.subheader("Simulate Customer Attributes & Score Churn Risk")
    st.markdown(
        "Configure customer contract parameters below to obtain instantaneous churn risk evaluation "
        "powered by the pre-trained Random Forest model."
    )

    pcol1, pcol2 = st.columns(2)

    with pcol1:
        input_tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=12)
        input_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=65.0, step=0.5)
        input_total = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(input_tenure * input_monthly))
        input_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with pcol2:
        input_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        input_internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        input_payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    if st.button("🚀 Evaluate Churn Risk Probability", type="primary"):
        if MODEL_PATH.exists():
            model = load_model(MODEL_PATH)

            # Format single customer row
            test_row = pd.DataFrame([{
                "tenure": input_tenure,
                "MonthlyCharges": input_monthly,
                "TotalCharges": input_total,
                "SeniorCitizen": input_senior,
                "Contract": input_contract,
                "InternetService": input_internet,
                "PaymentMethod": input_payment
            }])

            X_sim, _ = preprocess_data(test_row)
            prob = float(predict_probability(model, X_sim)[0])
            risk_pct = prob * 100

            st.divider()
            rcol1, rcol2 = st.columns([1, 2])

            with rcol1:
                st.metric(
                    label="Predicted Churn Probability",
                    value=f"{risk_pct:.1f}%",
                    delta=f"{'+' if risk_pct > 50 else '-'}{abs(risk_pct - 50):.1f}% vs baseline",
                    delta_color="inverse"
                )
                if risk_pct >= 60.0:
                    st.error("🚨 HIGH CHURN RISK")
                elif risk_pct >= 30.0:
                    st.warning("⚠️ MODERATE CHURN RISK")
                else:
                    st.success("✅ LOW CHURN RISK")

            with rcol2:
                st.markdown("#### Recommended Retention Actions")
                if risk_pct >= 60.0:
                    st.markdown("""
                    - **Urgent Contract Incentive:** Offer a 15% discount for upgrading from Month-to-month to an Annual contract.
                    - **Automated Billing Incentive:** Transition customer away from Electronic check to Automatic Card Payment with a one-time bill credit.
                    - **Dedicated Outreach:** Schedule a customer success check-in regarding Fiber optic service stability.
                    """)
                elif risk_pct >= 30.0:
                    st.markdown("""
                    - **Engagement Nudge:** Send curated feature discovery email highlighting untapped plan benefits.
                    - **Loyalty Program:** Enroll customer in anniversary loyalty rewards tier.
                    """)
                else:
                    st.markdown("""
                    - **Healthy Account:** Maintain current service cadence and invite to quarterly customer feedback survey.
                    """)
        else:
            st.error("Pre-trained model artifact not found. Please train the model using `python src/train_model.py`.")
