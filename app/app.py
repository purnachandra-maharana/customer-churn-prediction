import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Analysis Dashboard")

# =====================================================
# IMAGES FOLDER
# =====================================================

IMAGES_FOLDER = Path(
    r"E:\anaconda_projects\customer churn prediction\images"
)

IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Customer Churn CSV File",
    type=["csv"]
)

# =====================================================
# PROCESS FILE
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # -------------------------------------------------
    # DATASET INFO
    # -------------------------------------------------

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.write("### First 5 Rows")

    st.dataframe(df.head())

    # -------------------------------------------------
    # DATA CLEANING
    # -------------------------------------------------

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        df.dropna(inplace=True)

    # =================================================
    # CHART 1 : CHURN DISTRIBUTION
    # =================================================

    if "Churn" in df.columns:

        st.subheader("Churn Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.countplot(
            data=df,
            x="Churn",
            ax=ax
        )

        ax.set_title("Customer Churn Distribution")

        chart_path = IMAGES_FOLDER / "churn_distribution.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 2 : CONTRACT VS CHURN
    # =================================================

    if "Contract" in df.columns and "Churn" in df.columns:

        st.subheader("Contract Type vs Churn")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.countplot(
            data=df,
            x="Contract",
            hue="Churn",
            ax=ax
        )

        ax.set_title("Contract Type vs Churn")

        chart_path = IMAGES_FOLDER / "contract_vs_churn.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 3 : INTERNET SERVICE VS CHURN
    # =================================================

    if "InternetService" in df.columns and "Churn" in df.columns:

        st.subheader("Internet Service vs Churn")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.countplot(
            data=df,
            x="InternetService",
            hue="Churn",
            ax=ax
        )

        ax.set_title("Internet Service vs Churn")

        chart_path = IMAGES_FOLDER / "internet_vs_churn.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 4 : PAYMENT METHOD VS CHURN
    # =================================================

    if "PaymentMethod" in df.columns and "Churn" in df.columns:

        st.subheader("Payment Method vs Churn")

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.countplot(
            data=df,
            x="PaymentMethod",
            hue="Churn",
            ax=ax
        )

        plt.xticks(rotation=25)

        ax.set_title("Payment Method vs Churn")

        chart_path = IMAGES_FOLDER / "payment_vs_churn.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 5 : TENURE DISTRIBUTION
    # =================================================

    if "tenure" in df.columns:

        st.subheader("Tenure Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.histplot(
            df["tenure"],
            bins=30,
            kde=True,
            ax=ax
        )

        ax.set_title("Customer Tenure Distribution")

        chart_path = IMAGES_FOLDER / "tenure_distribution.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 6 : MONTHLY CHARGES VS CHURN
    # =================================================

    if "MonthlyCharges" in df.columns and "Churn" in df.columns:

        st.subheader("Monthly Charges vs Churn")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Churn",
            y="MonthlyCharges",
            ax=ax
        )

        ax.set_title("Monthly Charges vs Churn")

        chart_path = IMAGES_FOLDER / "monthlycharges_vs_churn.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    # =================================================
    # CHART 7 : TOTAL CHARGES VS CHURN
    # =================================================

    if "TotalCharges" in df.columns and "Churn" in df.columns:

        st.subheader("Total Charges vs Churn")

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Churn",
            y="TotalCharges",
            ax=ax
        )

        ax.set_title("Total Charges vs Churn")

        chart_path = IMAGES_FOLDER / "totalcharges_vs_churn.png"

        fig.savefig(
            chart_path,
            bbox_inches="tight"
        )

        st.pyplot(fig)

        st.success(
            f"Saved: {chart_path.name}"
        )

    st.success("All charts generated and saved successfully!")

else:

    st.info("Please upload a CSV file to begin analysis.")