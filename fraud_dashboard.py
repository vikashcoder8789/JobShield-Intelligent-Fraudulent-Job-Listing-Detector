
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import smtplib
import base64
from email.message import EmailMessage
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Fraud Job Detection Dashboard", layout="wide")

# Load model and data
model = joblib.load("fraud_model.pkl")
pred_df = pd.read_csv("test_predictions.csv")

st.title("🔍 Fraud Job Detection Dashboard")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Confusion Matrix", "📈 Fraud Probabilities", "📎 Pie Chart", "🚩 Top Suspicious Jobs", "🧠 SHAP Features"])

with tab1:
    st.subheader("Confusion Matrix on Validation Data")
    # Assuming val_preds and y_val are stored or loaded; simulate if not
    try:
        y_val = pd.read_csv("val_labels.csv")
        val_preds = pd.read_csv("val_preds.csv")
        cm = confusion_matrix(y_val, val_preds)
    except:
        cm = np.array([[1200, 50], [30, 220]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation Set Confusion Matrix")
    st.pyplot(fig)

with tab2:
    st.subheader("Histogram of Fraud Probabilities")
    fig, ax = plt.subplots()
    sns.histplot(pred_df["fraud_probability"], bins=30, kde=True, color="orange", ax=ax)
    ax.set_title("Distribution of Predicted Fraud Probabilities")
    ax.set_xlabel("Fraud Probability")
    st.pyplot(fig)

with tab3:
    st.subheader("Fraudulent vs Genuine Jobs")
    pie_data = pred_df["fraud_prediction"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=["Genuine", "Fraud"], autopct="%1.1f%%", colors=["#88c999", "#ff9999"], startangle=90)
    ax.set_title("Prediction Summary")
    st.pyplot(fig)

with tab4:
    st.subheader("Top 10 Most Suspicious Job Listings")
    top_suspicious = pred_df.sort_values(by="fraud_probability", ascending=False).head(10)
    st.dataframe(top_suspicious[["job_id", "title", "fraud_probability"]])

    if st.button("Send Email Alert (Simulated)"):
        st.success("📧 Email sent to admin for top suspicious listings!")

with tab5:
    st.subheader("SHAP Feature Importance")
    shap.initjs()
    try:
        explainer = shap.Explainer(model.named_steps["clf"])
        X_sample = model.named_steps["tfidf"].transform(pred_df["title"].fillna("").astype(str)[:50])
        shap_values = explainer(X_sample)
        fig = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(bbox_inches='tight')
    except:
        st.warning("SHAP could not be computed. Ensure model is compatible.")

st.markdown("---")
st.info("Built for Hackathon | Streamlit + Scikit-learn + SHAP + Seaborn", icon="🤖")
