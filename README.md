# Fraudulent-E-Commerce-Transactions
Unsupervised Fraud Detection in E-commerce Transactions  This project explores unsupervised learning techniques to detect fraudulent transactions in a simulated e-commerce dataset. The challenge: we don't rely on labeled fraud data during training — we try to "discover" the anomalies using Isolation Forest and Local Outlier Factor.

---

## 📂 Project Overview

**Objective**  
Detect potential fraudulent e-commerce transactions using unsupervised anomaly detection, and evaluate the performance using a known ground truth (labels are used *only for evaluation*, not for training).

**Dataset**  
- **Name:** Simulated E-commerce Fraud Dataset  
- **Source:** Kaggle  
- **Size:** ~1.47 million rows  
- **Label column:** `Is Fraudulent` (used *only* for evaluation)

---

## 🔍 Workflow Summary

### 1. 📊 Exploratory Data Analysis (EDA)
- Checked for null values, data types, and distributions
- Visualized transaction amount, quantity, customer age, and other features
- Detected imbalance: <5% of transactions are fraudulent

### 2. 🧪 Unsupervised Anomaly Detection

**Models used:**
- ✅ **Isolation Forest (ISF)**  
- ✅ **Local Outlier Factor (LOF)**

Each model predicts anomalies without using the target label.

### 3. 🎯 Threshold Tuning & ROC Analysis

- Anomaly scores from ISF converted to binary classification using threshold tuning
- ROC curve & AUC score used to evaluate model capability
- Several thresholds tested: `-0.1`, `-0.05`, `0.0`, `0.05`, etc.
- Confusion Matrix & Classification Report visualized for each threshold

---

## 📈 Results Summary

| Model          | ROC-AUC | Best Recall (Fraud) | Best Precision (Fraud) |
|----------------|---------|----------------------|-------------------------|
| Isolation Forest | ~0.56  | 0.49 (threshold -0.05) | 0.24 (threshold 0.0)   |
| Local Outlier Factor | ~0.50  | 0.02 | 0.10 |

🔎 **Observation:**
- ISF performs slightly better than random (ROC-AUC > 0.5), LOF performs near random.
- Very high recall leads to low precision (many false positives).
- Trade-off managed via threshold tuning.

---

## 📁 Files

| File | Description |
|------|-------------|
| `Fraudulent_E_Commerce_Transactions.ipynb` | Full notebook with EDA, modelling, and evaluation |
| `assets/roc_curve.png` | ROC curve for ISF |
| `assets/confusion_matrix_threshold0.png` | Confusion matrix (threshold 0.0) |
| `assets/confusion_matrix_threshold_-0.05.png` | Confusion matrix (threshold -0.05) |
| `requirements.txt` | List of required Python libraries |

---

## 🔧 Tech Stack

- Python 3.11
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🚧 Limitations & Next Steps

- Unsupervised methods have limited performance due to lack of supervision and class imbalance
- ROC-AUC only slightly above 0.5 → indicates low separation power
- Explore semi-supervised or deep learning (e.g. **Autoencoder**) methods
- Feature engineering or dimensionality reduction (e.g. PCA) could help

---

## 💡 Key Learnings

- Unsupervised fraud detection is **very hard**, especially with high class imbalance
- Threshold tuning and ROC curve are essential tools to evaluate anomaly detection
- Real-world fraud detection systems often require **hybrid approaches**

---

## 👨‍💻 Author

**[Your Name]**  
📧 your.email@example.com  
🌐 [LinkedIn/GitHub/Portfolio link]

---

> *This project is part of my Data Science Portfolio. Feel free to explore, fork, and reach out for collaboration or feedback!*
