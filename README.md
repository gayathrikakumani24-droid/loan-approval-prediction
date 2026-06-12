# 🏦 Smart Loan Approval Prediction System - LoanIQ

An AI-powered Loan Approval Prediction System built using **Machine Learning** and **Streamlit**. This application predicts whether a loan application is likely to be approved or rejected based on applicant details such as income, CIBIL score, loan amount, assets, education, and employment status.

The project also provides interactive visualizations and personalized loan suggestions for approved applicants.

---

## 📌 Features

### 🤖 Machine Learning Based Prediction

* Uses a Decision Tree Classifier.
* Handles imbalanced datasets through upsampling.
* Probability calibration using CalibratedClassifierCV.
* Predicts Loan Approval/Rejection with confidence score.

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
DecisionTreeClassifier
```

Parameters:

```python
DecisionTreeClassifier(
    criterion='entropy',
    class_weight='balanced',
    max_depth=6,
    min_samples_leaf=5,
    random_state=42
)
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

---

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

Add screenshot here:

```text
Demo/screenshots/dashboard.png
```

### Approval Result

```text
Demo/screenshots/approved.png
```

### Rejection Result

```text
Demo/screenshots/rejected.png
```

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

* Random Forest Classifier
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

