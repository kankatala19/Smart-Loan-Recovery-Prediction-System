# 🤖 Smart Loan Recovery System using Machine Learning

This project implements a smart loan recovery system that predicts the risk of non-repayment using machine learning and assigns tailored recovery strategies based on borrower behavior. It leverages data on missed payments, repayment history, income, and recovery status to help financial institutions improve recovery efficiency.


## 📝 Abstract

As the demand for consumer loans increases, financial institutions need smarter systems to identify high-risk loans and automate recovery strategies. This project applies Random Forest and clustering techniques to historical loan data to assess borrower risk. Based on the risk score, personalized recovery strategies such as legal action, settlements, or reminders are assigned, optimizing the recovery process.


## ⚠️ Problems in Existing Systems

1. Manual classification of risky borrowers is error-prone and inconsistent  
2. Delayed or inappropriate recovery strategies affect loan recovery success  
3. No use of predictive analytics in existing workflows  
4. High recovery costs due to inefficient targeting  
5. Limited insight into payment behavior and recovery trends


## ✅ Proposed System

1. **Data-Driven Risk Prediction** using Random Forest classifiers  
2. **Probability-Based Risk Scoring** using `predict_proba()`  
3. **Recovery Strategy Mapping** based on risk thresholds  
4. **EDA and Visual Analysis** to understand patterns in missed payments and income  
5. **Clustering** (e.g., K-Means) for borrower segmentation


## 🎯 Objectives

- Predict likelihood of loan recovery using historical features  
- Score borrowers using class 1 probability (non-recovery risk)  
- Classify borrowers into segments for targeted recovery  
- Visualize key patterns between payment history and recovery outcomes  
- Recommend automated actions for high- and low-risk cases


## 🌐 Project Domain

- **Domain**: FinTech, Credit Risk, Recovery Analytics  
- **Focus**: Intelligent classification of borrower recovery status and strategy recommendation


## 📋 Requirement Analysis

### Functional Requirements:

- Data loading and preprocessing (handling missing values, encoding)
- Feature selection: missed payments, income, loan amount, etc.
- Risk prediction using Random Forest
- Visualizations using Plotly
- Strategy mapping logic (threshold-based recovery plans)

### Non-Functional Requirements:

- Modular code structure  
- Reproducibility with fixed random seeds  
- Interpretability with feature importance  
- Visual appeal and interactive charts  
- Model flexibility (can switch to other classifiers)


## 🏗️ System Architecture

1. **Data Ingestion** from CSV  
2. **Feature Engineering** for key predictors (e.g., Num_Missed_Payments, Monthly_Income)  
3. **Risk Modeling** using Random Forest Classifier  
4. **Strategy Assignment** based on risk scores:  
   - `>0.75` → Legal Action  
   - `0.5–0.75` → Settlement Offer  
   - `<0.5` → Reminder  
5. **Visualization & Reporting** using Plotly/Seaborn


## 📊 Results & Analysis

- Clear pattern: More missed payments → higher risk of non-recovery  
- Income and repayment history strongly correlate with recovery outcomes  
- Risk scoring enables early intervention on likely defaulters  
- Strategy mapping simplifies collection process for financial teams  
- Balanced dataset with outlier handling and feature scaling


## 🧪 Example Outputs

- 📦 Box Plot: Missed Payments vs Recovery Status  
- 📈 Histogram: Loan Amount Distribution  
- 📊 Scatter Plot: Monthly Income vs Loan Risk  
- 🧮 Clustering: K-Means segments for borrower types  
- 📄 CSV Output: Predicted strategies and risk scores


## 📦 Tech Stack

- **Python**  
- **scikit-learn (Random Forest, KMeans)**  
- **Pandas, NumPy**  
- **Plotly, Seaborn**  
- **Jupyter Notebook / .py script**


## 🚀 Future Work

- Deploy strategy engine as API using Flask or FastAPI  
- Add Streamlit dashboard for loan managers  
- Track model drift using monthly datasets  
- Integrate with loan CRM systems  
- Real-time alerts for new defaulters


