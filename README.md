# Capstone Project: Predicting Policy Lapse (Churn Proxy)

## Project Overview
I work as an actuary in life insurance company, pricing lapse-supported life insurance products. Policy lapse is a major cost driver in insurance. Anticipating which customers are at high risk of lapse allows insurers to intervene with targeted retention actions.  Main research question of my project is: "Can machine learning classification models (logistic regression, decision trees, random forest, gradient boosting, and SVM) predict which policyholders are likely to lapse their life insurance policy within the next 12 months?". Because real lapse data is rarely available for open-source projects, this capstone **reframes the Telco Customer Churn dataset** as a **proxy for insurance lapse**. Churn = lapse.

**Business Question:**  
Can we use ML to predict lapse risk, segment policyholders into **risk deciles**, and generate actionable insights for retention?

# Note
Although the Telco Customer Churn dataset already appears in the syllabus for basic classification exercises, this project uses it in a non-standard, domain-specific way. Instead of stopping at model accuracy, I reframed churn as an insurance lapse analogue, applied threshold tuning, calibration, and risk-decile segmentation, and translated results into actionable business implications for lapse-sensitive products. This shows how even a familiar dataset can generate new insights and practical value when aligned with real industry contexts.

## Files

- [Jupyter Notebook — `capstone_lapse_eda.ipynb`](capstone_lapse_final_ok.ipynb)
- [Dataset - 'WA_Fn-UseC_-Telco-Customer-Churn.csv'](WA_Fn-UseC_-Telco-Customer-Churn.csv)

---

## Data & Methods
- **Dataset:** Telco Customer Churn (7,043 rows, 21 features).  
- **Target:** `Churn` → treated as lapse indicator (1 = lapse, 0 = retain).  
- **Features:** Demographics, service usage, billing info.  
- **Models tested:** Logistic Regression, Decision Tree, KNN, SVC.  
- **Process:**  
  1. Data cleaning and preprocessing (numeric scaling + categorical one-hot encoding).  
  2. Train/test split with stratification.  
  3. Cross-validation and **GridSearch** hyperparameter tuning.  
  4. Model selection and evaluation (ROC-AUC, PR-AUC, Accuracy, F1).  
  5. Risk-decile segmentation (D1–D10).  

---

## Results
- **Best Model:** Logistic Regression provided the best balance of interpretability and predictive performance.  
- **Performance (test set):** ROC-AUC ≈ 0.835, PR-AUC ≈ 0.617, Accuracy ≈ 0.80, F1 ≈ 0.603.  
- **Risk Deciles:** Customers segmented into 10 groups (D1–D10).  
  - D1 = safest (lapse rate < 1%).  
  - D10 = riskiest (lapse rate > 69%).  
- **Business Impact:** Top deciles (D9–D10) are clear targets for retention campaigns, while lower deciles can be deprioritized.  

---

## Key Findings
- Even though the Telco Churn dataset is widely used for basic ML classification, this project demonstrates a **non-standard application**: repurposing churn → lapse.  
- **Risk decile segmentation** enables actionable business strategies beyond raw model accuracy.  
- The model’s **probability outputs** can be translated into actionable thresholds for retention decisions.  

---

## Next Steps
1. Apply this framework to **actual insurance lapse data**.  
2. Explore **ensemble models** (XGBoost, Random Forests) with explainability tools (e.g., SHAP).  
3. Conduct **cost–benefit analysis** of retention offers (intervention cost vs. lifetime value saved).  
4. Deploy as a batch-scoring workflow with scheduled decile risk reports.  

---

## Link to Jupyter Notebook
➡️ [Capstone Final Notebook](notebooks/capstone_lapse_final.ipynb)
