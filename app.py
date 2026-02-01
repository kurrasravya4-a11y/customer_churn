import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Analytics & Prediction", layout="wide")

# =========================
# Custom CSS Styling
# =========================
st.markdown("""
<style>
.main {background-color: #f7f9fc;}
h1 {color: #0f172a; font-weight: 800;}
h2, h3 {color: #1e293b;}

.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 16px;
    color: #64748b;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #0f172a;
}

.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Files
# =========================
df = pd.read_csv("Telco_customer_churn.csv")
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
model_columns = joblib.load("model_columns.pkl")
cat_cols = joblib.load("cat_cols.pkl")
num_cols = joblib.load("num_cols.pkl")
metrics_df = pd.read_csv("model_metrics.csv")

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["Overview", "EDA", "Model Metrics", "Prediction"])

st.title("📊 Telecom Customer Churn Analysis & Prediction System")

# =========================
# OVERVIEW PAGE
# =========================
if page == "Overview":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    To predict whether a telecom customer is likely to churn (leave the service) using Machine Learning.

    **Problem Type:**  
    Binary Classification (Churn: Yes / No)

    **Final Model Used:**  
    Optimized Random Forest Classifier (with SMOTE & GridSearchCV)
    """)

    clean_df = df.drop(columns=["Churn Reason"])
    clean_df["Total Charges"] = pd.to_numeric(clean_df["Total Charges"], errors="coerce")
    clean_df["Total Charges"] = clean_df["Total Charges"].fillna(clean_df["Total Charges"].median())

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Rows</div>
            <div class="metric-value">{df.shape[0]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Total Columns</div>
            <div class="metric-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">Missing Values (After Cleaning)</div>
            <div class="metric-value">{int(clean_df.isnull().sum().sum())}</div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📄 Top 20 Rows Preview")
    st.dataframe(df.head(20))

    st.subheader("📈 Summary Statistics")
    st.dataframe(clean_df.describe())

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# EDA PAGE
# =========================
elif page == "EDA":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("📈 Exploratory Data Analysis (EDA)")

    churn_counts = df["Churn Label"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(churn_counts.index, churn_counts.values)
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Churn Label", y="Tenure Months", data=df, ax=ax2)
    ax2.set_title("Tenure vs Churn")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn Label", y="Monthly Charges", data=df, ax=ax3)
    ax3.set_title("Monthly Charges vs Churn")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots()
    sns.countplot(x="Contract", hue="Churn Label", data=df, ax=ax4)
    ax4.set_title("Contract Type vs Churn")
    plt.xticks(rotation=30)
    st.pyplot(fig4)

    corr_df = df[["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", "Churn Value"]].copy()
    corr_df["Total Charges"] = pd.to_numeric(corr_df["Total Charges"], errors="coerce")

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax5)
    ax5.set_title("Correlation Heatmap")
    st.pyplot(fig5)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL METRICS PAGE
# =========================
elif page == "Model Metrics":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🤖 Model Performance & Comparison")

    comparison_df = metrics_df.copy()
    if "Model" in comparison_df.columns:
        comparison_df.set_index("Model", inplace=True)

    st.dataframe(comparison_df.round(4), use_container_width=True)

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png")

    st.subheader("ROC Curve")
    st.image("roc_curve.png")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION PAGE
# =========================
elif page == "Prediction":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("🔮 Customer Churn Prediction")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure Months", 0, 100, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 500.0, 70.0)
        total = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict Churn"):

        input_dict = {col: 0 if col in num_cols else "Unknown" for col in model_columns}

        input_dict["Tenure Months"] = tenure
        input_dict["Monthly Charges"] = monthly
        input_dict["Total Charges"] = total
        input_dict["Contract"] = contract
        input_dict["Internet Service"] = internet

        input_df = pd.DataFrame([input_dict])
        input_processed = preprocessor.transform(input_df)

        pred = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        st.subheader("📊 Prediction Result")

        st.progress(int(prob * 100))
        st.caption(f"Churn Probability: {prob*100:.2f}%")

        if pred == 1:
            st.error(f"⚠️ Customer is likely to CHURN")
        else:
            st.success(f"✅ Customer is likely to STAY")

    st.markdown('</div>', unsafe_allow_html=True)
