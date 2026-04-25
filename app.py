"""
Credit Card Churn — Streamlit POC
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Credit Card Churn Predictor", page_icon="💳", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("💳 Credit Card Customer Churn Predictor")
st.caption("Thera Bank POC — Random Forest classifier deployed on AWS (S3 + EC2)")
st.markdown("---")

st.subheader("Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    customer_age = st.number_input("Customer Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["F", "M"])
    dependent_count = st.number_input("Dependents", 0, 10, 2)
    education = st.selectbox("Education Level",
        ["College", "Doctorate", "Graduate", "High School", "Post-Graduate", "Uneducated", "Unknown"])
    marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single", "Unknown"])
    income = st.selectbox("Income Category",
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])

with col2:
    st.markdown("**Account Activity**")
    card_category = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
    months_on_book = st.number_input("Months on Book", 0, 60, 36)
    total_relationship_count = st.number_input("Products Held", 1, 6, 3)
    months_inactive = st.number_input("Months Inactive (last 12)", 0, 12, 2)
    contacts_count = st.number_input("Contacts (last 12 months)", 0, 10, 2)

with col3:
    st.markdown("**Financials**")
    credit_limit = st.number_input("Credit Limit ($)", 0.0, value=8000.0, step=100.0)
    total_revolving_bal = st.number_input("Revolving Balance ($)", 0.0, value=1200.0, step=50.0)
    avg_open_to_buy = st.number_input("Avg Open to Buy ($)", 0.0, value=6800.0, step=100.0)
    total_amt_chng = st.number_input("Amt Change Q4/Q1", 0.0, value=0.75, step=0.01)
    total_trans_amt = st.number_input("Total Transaction Amt ($)", 0.0, value=4400.0, step=50.0)
    total_trans_ct = st.number_input("Total Transaction Count", 0, value=65, step=1)
    total_ct_chng = st.number_input("Count Change Q4/Q1", 0.0, value=0.7, step=0.01)
    avg_util = st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.15, step=0.01)

st.markdown("---")

if st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True):
    raw = {
        "Customer_Age": customer_age, "Dependent_count": dependent_count,
        "Months_on_book": months_on_book, "Total_Relationship_Count": total_relationship_count,
        "Months_Inactive_12_mon": months_inactive, "Contacts_Count_12_mon": contacts_count,
        "Credit_Limit": credit_limit, "Total_Revolving_Bal": total_revolving_bal,
        "Avg_Open_To_Buy": avg_open_to_buy, "Total_Amt_Chng_Q4_Q1": total_amt_chng,
        "Total_Trans_Amt": total_trans_amt, "Total_Trans_Ct": total_trans_ct,
        "Total_Ct_Chng_Q4_Q1": total_ct_chng, "Avg_Utilization_Ratio": avg_util,
        "Gender": gender, "Education_Level": education, "Marital_Status": marital,
        "Income_Category": income, "Card_Category": card_category,
    }
    df = pd.DataFrame([raw])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)
    X_scaled = scaler.transform(df_aligned)
    prob_churn = float(model.predict_proba(X_scaled)[0, 1])
    pred = int(prob_churn >= 0.5)

    st.markdown("## Prediction Result")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric("Churn Probability", f"{prob_churn:.1%}",
                  delta=f"{'High risk' if prob_churn >= 0.5 else 'Low risk'}",
                  delta_color="inverse")
    with res_col2:
        if prob_churn >= 0.7:
            st.error(f"🚨 **HIGH RISK** ({prob_churn:.1%}) — Recommend immediate retention action: "
                     "priority outreach, targeted offer, or relationship manager contact.")
        elif prob_churn >= 0.4:
            st.warning(f"⚠️ **MEDIUM RISK** ({prob_churn:.1%}) — Monitor closely. "
                       "Consider engagement campaign or fee waiver evaluation.")
        else:
            st.success(f"✅ **LOW RISK** ({prob_churn:.1%}) — Customer likely to stay. "
                       "Standard relationship management sufficient.")
    st.progress(prob_churn)
    st.caption(f"Raw churn probability: {prob_churn:.4f} | Threshold: 0.50 | "
               f"Predicted: {'Attrited (1)' if pred == 1 else 'Existing (0)'}")

st.markdown("---")
with st.expander("ℹ️ About this POC"):
    st.markdown("""
    **Model**: Random Forest (300 trees), trained with SMOTE-balanced data.
    **Test set**: Recall 0.865, Precision 0.865, F1 0.865, ROC-AUC 0.989.
    **Infrastructure**: Model artifacts in S3, Streamlit on EC2 t3.small (Ubuntu 22.04).
    **Intended user**: Bank retention analysts & relationship managers.
    """)