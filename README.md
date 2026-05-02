# 💳 Credit Card Fraud Detection System

## 📌 Overview
This project builds an end-to-end Machine Learning system to detect fraudulent credit card transactions using real-world data and imbalanced classification techniques.

---

## 🎯 Problem Statement
Credit card fraud causes significant financial losses. Detecting fraud in real-time is challenging due to:
- Highly imbalanced data
- Evolving fraud patterns
- Need for low false positives

---

## 💡 Solution
- Applied **SMOTE** to handle class imbalance  
- Trained **Random Forest model**  
- Evaluated using **Precision, Recall, Confusion Matrix**  
- Built a **prediction system for new transactions**

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Imbalanced-learn  

---

## 📊 Results

| Metric | Value |
|-------|------|
| Recall (Fraud Detection) | ~80% |
| Precision | ~87% |
| False Positives | Very Low |

---

## 📸 Visualizations

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

---

## ▶️ How to Run

```bash
git clone <your-repo-link>
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
python main.py
🧠 Learnings
Handling imbalanced datasets
Feature scaling & preprocessing
Model evaluation techniques
Real-world ML pipeline design
# 📸 3️⃣ ADD IMAGES (IMPORTANT)

Make sure this exists:

```text id="img1"
images/
 ├── dataset_preview.png
 ├── preprocessing_output.png
 ├── smote_output.png
 └── confusion_matrix.png
 ## 🚀 API Usage

Run:
uvicorn api.app:app --reload

Test:
http://127.0.0.1:8000/docs

🚀 Author

Arpita Bhendigeri


---
