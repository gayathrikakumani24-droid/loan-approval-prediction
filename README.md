# 🏦 Smart Loan Approval Prediction System - LoanIQ

🌐 Live Demo: https://loaniq-assessment.streamlit.app/

An AI-powered Loan Approval Prediction System built using Random Forest Machine Learning and Streamlit. The system evaluates applicant financial profiles, credit scores, and asset information to predict loan approval decisions with confidence scores and explainable risk indicators.

The project also provides interactive visualizations and personalized loan suggestions for approved applicants.

---

## 📌 Features

### 🤖 Machine Learning Based Prediction

* Uses a Random Forest Classifier for robust and accurate predictions.
* Handles imbalanced datasets through upsampling.
* Probability calibration using CalibratedClassifierCV.
* Predicts Loan Approval/Rejection with confidence score.
* Reduces overfitting compared to a single Decision Tree.


### 📊 Data Visualization

* Loan Approval Distribution
* Education vs Loan Status
* Income Distribution
* CIBIL Score Analysis
* Asset Distribution
* Feature Importance Analysis

### 🏦 Banking Dashboard UI

* Modern banking-inspired theme
* Interactive form inputs
* Color-coded approval/rejection results
* Suggested loan recommendations
* Responsive Streamlit interface

### 💡 Business Rule Engine

Additional approval checks based on:

* CIBIL Score
* Annual Income
* Total Assets
* Loan Amount

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Backend Development       |
| Pandas       | Data Processing           |
| NumPy        | Numerical Operations      |
| Scikit-Learn | Machine Learning          |
| Matplotlib   | Data Visualization        |
| Seaborn      | Statistical Visualization |
| Streamlit    | Frontend Dashboard        |

---

# 📂 Project Structure

```text
Loan_Approval_Project/
│
├── app.py
├── train_model.py
├── loan_approval_dataset.csv
├── requirements.txt
├── README.md
├── trained_model.pkl
├── feature_columns.pkl
│
├── Demo/
│   ├── screenshots
│


```

---

# 📈 Machine Learning Workflow

### 1️⃣ Data Collection

The dataset contains:

* Number of Dependents
* Education
* Self Employed Status
* Annual Income
* Loan Amount
* Loan Term
* CIBIL Score
* Residential Assets
* Commercial Assets
* Luxury Assets
* Bank Assets
* Loan Status

---

### 2️⃣ Data Preprocessing

Performed the following operations:

* Removed unwanted spaces from column names
* Label Encoding of categorical variables
* Feature Engineering
* Asset Aggregation
* Dataset Balancing

Example:

```python
df['total_assets'] = (
    df['residential_assets_value']
    + df['commercial_assets_value']
    + df['luxury_assets_value']
    + df['bank_asset_value']
)
```

---

### 3️⃣ Dataset Balancing

Loan datasets are often imbalanced.

Upsampling was applied:

```python
from sklearn.utils import resample
```

Benefits:

* Reduces bias
* Improves recall
* Better approval/rejection predictions

---

### 4️⃣ Model Training

Model Used:

```python
RandomForestClassifier
```

Parameters:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)
```

Why Random Forest?

* Combines multiple decision trees for improved accuracy.
* Reduces overfitting.
* Provides more stable predictions.
* Utilizes multiple applicant features instead of relying heavily on a single feature.
* Produces better generalization on unseen data.

```
```


---

### 5️⃣ Probability Calibration

```python
CalibratedClassifierCV
```

Purpose:

* Provides realistic confidence scores.
* Reduces overconfident predictions.

---

# 🎯 Prediction Logic

The model predicts:

```text
0 → Approved
1 → Rejected
```

After prediction:

```python
prediction = model.predict(user_input)[0]
```

The application displays:

* Loan Status
* Confidence Score
* Suggested Loan Types
* Rejection Reasons (if rejected)

---

# 🏦 Suggested Loan Types

Based on user profile:

| Condition        | Suggested Loan |
| ---------------- | -------------- |
| Loan ≤ ₹20 Lakhs | Personal Loan  |
| Loan > ₹20 Lakhs | Home Loan      |
| Self Employed    | Business Loan  |
| Student Profile  | Education Loan |

---

# 📊 Visualizations Included

### Loan Approval Distribution

Shows percentage of:

* Approved Loans
* Rejected Loans

---

### CIBIL Score Distribution

Displays relationship between:

* Credit score
* Loan approval probability

---

### Income Distribution

Analyzes:

* Annual income
* Approval trends

---

### Feature Importance

Shows most influential features:

* CIBIL Score
* Income
* Loan Amount
* Assets

# 📊 Model Performance

The LoanIQ system uses a **Random Forest Classifier** combined with **CalibratedClassifierCV** to provide accurate and reliable loan approval predictions.

### Evaluation Results

| Metric    | Score              |
| --------- | ------------------ |
| Accuracy  | **97.83%**         |
| Precision | **99% (Approved)** |
| Precision | **97% (Rejected)** |
| Recall    | **97% (Approved)** |
| Recall    | **99% (Rejected)** |
| F1-Score  | **98%**            |

### Classification Report

```text
Accuracy: 97.83%

              precision    recall  f1-score   support

Approved(0)      0.99      0.97      0.98       323
Rejected(1)      0.97      0.99      0.98       323

accuracy                              0.98       646
macro avg         0.98      0.98      0.98       646
weighted avg      0.98      0.98      0.98       646
```

### Feature Importance Analysis

The Random Forest model identified the following features as the most influential factors in loan approval decisions:

| Feature            | Importance |
| ------------------ | ---------- |
| CIBIL Score        | 86.53%     |
| Loan Term          | 4.77%      |
| Loan Amount        | 1.84%      |
| Total Assets       | 1.18%      |
| Commercial Assets  | 1.11%      |
| Residential Assets | 1.08%      |
| Luxury Assets      | 1.07%      |
| Bank Assets        | 0.97%      |
| Annual Income      | 0.93%      |

### Performance Summary

* Achieved **97.83% prediction accuracy** on the test dataset.
* Demonstrates strong classification capability for both approved and rejected loan applications.
* Utilizes probability calibration for more reliable confidence scores.
* Provides explainable predictions through feature importance analysis.
* Suitable for educational demonstrations, machine learning projects, and portfolio showcases.

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/loan-approval-prediction.git
```

```bash
cd loan-approval-prediction
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

Application opens automatically at:

```text
http://localhost:8501
```

---

# 📸 Screenshots

### Dashboard

<img width="1366" height="647" alt="Screenshot (316)" src="https://github.com/user-attachments/assets/b5381aad-a393-43a5-b6d8-d976331ff5ef" />


### Approval Result

<img width="1366" height="641" alt="Screenshot (319)" src="https://github.com/user-attachments/assets/602cb693-d3fb-43ec-aecc-25c4fcd24d33" />

### Rejection Result

<img width="1366" height="652" alt="Screenshot (320)" src="https://github.com/user-attachments/assets/4a2e8444-a8d4-43b7-a644-1ca2d36901b9" />


### Dataset Analysis

<img width="1366" height="640" alt="Screenshot (322)" src="https://github.com/user-attachments/assets/16c4a01c-102f-4ab3-b553-9d1310e76f89" />



---

# 📋 Example Input

| Feature       | Value      |
| ------------- | ---------- |
| Dependents    | 1          |
| Education     | Graduate   |
| Self Employed | No         |
| Annual Income | ₹15,00,000 |
| Loan Amount   | ₹5,00,000  |
| CIBIL Score   | 800        |
| Total Assets  | ₹8,00,000  |

### Prediction

```text
Loan Approved
Confidence: 84.18%
```

---

# 🔮 Future Enhancements

* XGBoost Integration
* SHAP Explainability
* PDF Loan Report Generation
* User Authentication
* Cloud Deployment
* Model Monitoring Dashboard
* Loan EMI Calculator

---

# 🎓 Academic Relevance

This project demonstrates concepts from:

* Machine Learning
* Data Preprocessing
* Feature Engineering
* Classification Algorithms
* Model Evaluation
* Data Visualization
* Web Application Development

Suitable for:

* Mini Projects
* Major Projects
* College Demonstrations
* Machine Learning Portfolio Projects

---

# 👨‍💻 Author

**Gayathri**

B.Tech Computer Science Engineering (AI & ML)

---

# 📜 License

This project is developed for educational and learning purposes.

MIT License © 2026

---

⭐ If you found this project useful, consider giving it a star on GitHub.

