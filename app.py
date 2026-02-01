import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Telecom AI Churn Studio", layout="wide")

# =========================
# Custom Dark Theme Styling
# =========================
st.markdown("""
<style>
body {background-color: #0f172a; color: white;}
.main {background-color: #0f172a;}
h1, h2, h3 {color: #38bdf8;}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    font-weight: bold;
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
num_cols = joblib.load("num_cols.pkl")
metrics_df = pd.read_csv("model_metrics.csv")

st.title("🚀 Telecom AI Churn Studio")

menu = st.sidebar.radio(
    "📌 Navigation",
    ["📊 Dashboard",
     "📈 Data Insights",
     "🤖 Model Evaluation",
     "🔮 Smart Prediction"]
)

# =========================
# DASHBOARD PAGE
# =========================
if menu == "📊 Dashboard":

    st.subheader("📌 Business Overview")

    total_customers = df.shape[0]
    churn_rate = round((df["Churn Value"].mean()) * 100, 2)
    avg_monthly = round(df["Monthly Charges"].mean(), 2)

    col1, col2, col3 = st.columns(3)

    col1.metric("👥 Total Customers", total_customers)
    col2.metric("📉 Churn Rate", f"{churn_rate}%")
    col3.metric("💰 Avg Monthly Charges", f"${avg_monthly}")

    st.markdown("---")

    # Customer Distribution
    st.subheader("📊 Customer Distribution")

    fig = px.pie(
        df,
        names="Churn Label",
        hole=0.5,
        color_discrete_sequence=["#38bdf8", "#f87171"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Statistical Summary
    st.subheader("📈 Statistical Summary")

    numeric_df = df.select_dtypes(include=["int64", "float64"])
    st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.markdown("---")

    # Dataset Preview
    st.subheader("📄 Sample Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


# =========================
# DATA INSIGHTS PAGE
# =========================
elif menu == "📈 Data Insights":

    st.subheader("Tenure Impact on Churn")
    fig1 = px.box(
        df,
        x="Churn Label",
        y="Tenure Months",
        color="Churn Label",
        color_discrete_sequence=["#38bdf8", "#f87171"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Monthly Charges Distribution")
    fig2 = px.histogram(
        df,
        x="Monthly Charges",
        color="Churn Label",
        barmode="overlay",
        color_discrete_sequence=["#38bdf8", "#f87171"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Contract Type vs Churn")
    fig3 = px.histogram(
        df,
        x="Contract",
        color="Churn Label",
        barmode="group",
        color_discrete_sequence=["#38bdf8", "#f87171"]
    )
    st.plotly_chart(fig3, use_container_width=True)


# =========================
# MODEL EVALUATION PAGE
# =========================
elif menu == "🤖 Model Evaluation":

    st.subheader("📊 Model Comparison Metrics")
    st.dataframe(metrics_df.round(4), use_container_width=True)

    st.subheader("Confusion Matrix")
    st.image("confusion_matrix.png", use_column_width=True)

    st.subheader("ROC Curve")
    st.image("roc_curve.png", use_column_width=True)


# =========================
# SMART PREDICTION PAGE
# =========================
elif menu == "🔮 Smart Prediction":

    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure Months", 0, 100, 12)
        monthly = st.slider("Monthly Charges", 0, 500, 70)
        total = st.number_input("Total Charges", 0.0, 20000.0, 1000.0)

    with col2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    if st.button("Analyze Customer"):

        input_dict = {
            col: 0 if col in num_cols else "Unknown"
            for col in model_columns
        }

        input_dict["Tenure Months"] = tenure
        input_dict["Monthly Charges"] = monthly
        input_dict["Total Charges"] = total
        input_dict["Contract"] = contract
        input_dict["Internet Service"] = internet

        input_df = pd.DataFrame([input_dict])
        processed = preprocessor.transform(input_df)

        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        st.subheader("📊 Prediction Result")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {
                    'color': "#f87171" if prob > 0.5 else "#38bdf8"
                }
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            st.error("⚠️ High Risk Customer – Likely to Churn")
        else:
            st.success("✅ Customer Retention Likely")
