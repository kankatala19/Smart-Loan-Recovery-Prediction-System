import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load model and features
model = pickle.load(open("rf_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(page_title="Smart Loan Recovery Predictor", layout="wide")
st.title("💼 Smart Loan Recovery Predictor")

st.markdown("""
This tool predicts whether a borrower is **high risk** or **low risk**, and suggests a suitable recovery strategy.
Upload your dataset with the correct feature columns, or manually enter borrower details.
""")

# Sidebar to toggle input mode
option = st.sidebar.radio("Select Mode", ["📂 Upload CSV", "✍️ Manual Entry"])

# CSV File Upload
if option == "📂 Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload Borrower CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        try:
            input_data = df[features]

            # Predict risk scores
            risk_scores = model.predict_proba(input_data)[:, 1]
            df['Risk_Score'] = risk_scores
            df['Predicted_High_Risk'] = (risk_scores > 0.5).astype(int)

            # Assign strategy
            def assign_strategy(score):
                if score > 0.75:
                    return "⚠️ Legal Action"
                elif score > 0.5:
                    return "💬 Settlement Plan"
                else:
                    return "📩 Reminder"

            df['Strategy'] = df['Risk_Score'].apply(assign_strategy)

            # Display results with full-width table
            st.subheader("🔍 Prediction Results")
            display_cols = ['Risk_Score', 'Predicted_High_Risk', 'Strategy'] + [
                col for col in df.columns if col not in features + ['Risk_Score', 'Predicted_High_Risk', 'Strategy']
            ]
            st.dataframe(df[display_cols], use_container_width=True)

            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "loan_predictions.csv", "text/csv")

            # Feature importance
            st.subheader("📊 Feature Importance")
            importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values(by="Importance", ascending=True)
            fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Model Feature Importance")
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.warning("Please make sure your CSV includes these columns:")
            st.code(", ".join(features))

# Manual Entry
elif option == "✍️ Manual Entry":
    st.subheader("Enter Borrower Details")

    manual_input = {feature: st.number_input(f"{feature}", value=0.0) for feature in features}

    if st.button("Predict"):
        input_df = pd.DataFrame([manual_input])
        prob = model.predict_proba(input_df)[0, 1]
        prediction = "High Risk" if prob > 0.5 else "Low Risk"

        if prob > 0.75:
            strategy = "⚠️ Legal Action"
        elif prob > 0.5:
            strategy = "💬 Settlement Plan"
        else:
            strategy = "📩 Reminder"

        st.success(f"Risk Score: `{prob:.2f}` → **{prediction}**")
        st.info(f"Recommended Strategy: {strategy}")
