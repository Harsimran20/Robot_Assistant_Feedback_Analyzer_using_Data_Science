# 🤖 AI Robotics Feedback Classifier

This project uses **Data Science** and **Machine Learning** to classify user feedback and predict whether a robotic task was successfully completed. It applies Natural Language Processing (NLP) techniques and a Multinomial Naive Bayes classifier to analyze textual feedback.

---

## 📌 Problem Statement

In robotics environments, user feedback helps assess task performance. Manually evaluating this feedback is time-consuming and subjective. This project automates the classification of such feedback into binary outcomes:
- `1`: Task Success
- `0`: Task Failure

---

## 🧰 Tools & Technologies

- **Python**
- **Pandas** – for data manipulation
- **Scikit-learn** – ML models, training, evaluation
- **TfidfVectorizer / CountVectorizer** – for feature extraction
- **Jupyter Notebook** – for prototyping

---

## 🗃️ Dataset Structure

The dataset (e.g., `feedback_data.csv`) includes:

| Column              | Description                         |
|---------------------|-------------------------------------|
| `user_feedback_text`| Textual feedback from the user      |
| `task_success`      | Binary target (0 = fail, 1 = success)|

---

## 🚀 How to Run

### 1. Clone the repository

git clone repository
cd ai-robotics-feedback-classifier

### 2. Install dependencies
pip install -r requirements.txt

### 3. Launch Jupyter Notebook
jupyter notebook
Open notebook/feedback_classifier.ipynb to run and test the model.

💡 Sample Code
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("data/feedback_data.csv")

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['user_feedback_text'])
y = df['task_success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
📈 Example Output

              precision    recall  f1-score   support

           0       0.82      0.75      0.78       100
           1       0.85      0.90      0.87       150

    accuracy                           0.84       250
   macro avg       0.83      0.82      0.83       250
weighted avg       0.84      0.84      0.84       250
🧠 What This Report Means
Precision: Accuracy of positive predictions (e.g., how often predicted "success" is actually success).

Recall: Ability to find all positive cases (sensitivity).

F1-score: Balance between precision and recall.

Support: Number of actual instances per class.

