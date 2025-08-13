# Disease Classification using Apache Spark (Scala)

This project implements a large-scale disease classification system using **Apache Spark** with **Scala**, processing over **246,000 records** and **376 features** from a Kaggle dataset of diseases and symptoms.  
It demonstrates end-to-end **ETL**, **feature engineering**, and **machine learning model training** on distributed data.

---

## 📌 Project Overview
- **Dataset**: [Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)
- **Data Size**: ~246k rows, 376 symptom features, 773 unique diseases (top 10 used for classification)
- **Goal**: Predict the disease category based on given symptoms using Spark MLlib.
- **Tech Stack**:
  - **Apache Spark** (DataFrames, MLlib)
  - **Scala**
  - **Kaggle Dataset**
  - **Jupyter / IntelliJ / Databricks (optional)**

---

## ⚙️ Features
- Data ingestion from CSV using Spark.
- Data cleaning & preprocessing.
- Symptom encoding for model training.
- Feature selection for dimensionality reduction.
- Model training and evaluation using:
  - Logistic Regression
  - Naïve Bayes
  - Decision Tree Classifier
- Performance comparison using **Accuracy, Precision, Recall, F1-score, and ROC-AUC**.

---

## 📊 Dataset Details
- **773 diseases** and **377 symptoms** in the raw dataset.
- Filtered to **top 10 diseases** with the most records for classification.
- Artificially generated dataset with preserved **Symptom Severity** and **Disease Occurrence Probability**.

---

## 🧪 Results
| Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.0904   | 0.0081    | 0.0900 | 0.1500   | 0.5000  |
| Decision Tree      | 0.3374   | 0.5055    | 0.3374 | 0.3448   | 0.8014  |
| **Naïve Bayes**    | **0.9804** | **0.9806** | **0.9804** | **0.9803** | **0.9997** |

**Best Model:** Naïve Bayes  
- Highest accuracy, precision, recall, and F1-score.
- ROC-AUC close to 1, indicating excellent classification performance.

---

## 🛠 Installation & Usage
### Prerequisites
- Apache Spark 3.x
- Scala 2.12+
- sbt (Scala Build Tool)

### Clone the Repository
```bash
git clone https://github.com/your-username/disease-classification-spark-scala.git
cd disease-classification-spark-scala
