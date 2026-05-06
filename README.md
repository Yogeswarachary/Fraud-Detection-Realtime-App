# 🚀 Real-Time Fraud Detection System

A production-ready **Machine Learning Fraud Detection System** built using **FastAPI (Backend)** and **Streamlit (Frontend)**.

This project detects fraudulent financial transactions in real-time using multiple ML models with explainability.

---

## 📌 Features

* 🔍 Real-time fraud prediction
* 📊 Multiple ML models:

  * CatBoost
  * Balanced Random Forest
  * Hybrid CatBoost
  * Hybrid Logistic Regression
* 🧠 Explainable AI (Top reasons for prediction)
* 📁 Bulk CSV upload support
* ⚡ FastAPI backend for high performance
* 🎯 Streamlit UI for easy interaction

---

## 🏗️ Project Structure

```
fraud-detection-realtime-app/
│
├── backend/
│   ├── models/
│   ├── main.py
│   ├── custom_models.py
│
├── frontend/
│   ├── app.py
│
├── sample_data/
│   ├── sample_data.csv
│   ├── sample_data.json
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Yogeswarachary/fraud-detection-realtime-app.git
cd fraud-detection-realtime-app
```

---

### 2️⃣ Create Virtual Environment

```bash
conda create -n fraud_env_312 python=3.12
conda activate fraud_env_312
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Backend (FastAPI)

```bash
cd backend
fastapi dev main.py
```

👉 API will run at:
`http://127.0.0.1:8000`

---

### 5️⃣ Run Frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

---

## 📊 Sample Input

You can test using:

* `sample_data/sample_data.csv`
* `sample_data/sample_data.json`

---

## 🧠 Models Used

| Model           | Description                       |
| --------------- | --------------------------------- |
| CatBoost        | Gradient boosting model           |
| Balanced RF     | Handles class imbalance           |
| Hybrid CatBoost | Ensemble model                    |
| Hybrid LR       | Logistic regression with stacking |

---

## 🔍 Explainability

* CatBoost → SHAP-based explanations
* Other models → Perturbation-based feature importance

---

## 🚀 Deployment

* Backend → Render / Railway
* Frontend → Streamlit Cloud

---

## 👨‍💻 Author

**Yogeswarachary Modepalli**

* Aspiring Data Scientist
* Skilled in ML, FastAPI, and real-time systems

---

## ⭐ Future Improvements

* Model monitoring
* API authentication
* Docker support
* CI/CD pipeline

---

## 📌 Note

This project demonstrates **end-to-end ML deployment**, including:

* Model building
* API development
* Frontend integration
* Real-time prediction

##### ⚠️ Note: Initial request may take a few seconds due to model loading on free-tier hosting (Render).