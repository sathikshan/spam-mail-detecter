# spam-mail-detecter
# Step 1: Install Required Libraries
!pip install pandas numpy scikit-learn

# Step 2: Import Libraries and Load the Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from google.colab import files

# Step 3: Upload the Dataset
uploaded = files.upload()

# Step 4: Load the Dataset
df = pd.read_csv('/content/spam (1).csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Assuming the columns are named 'v1' for labels and 'v2' for messages
df.columns = ['label', 'message']  # Renaming columns for clarity

# Step 5: Preprocess the Data
# Encode labels: spam as 1 and legitimate as 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Check for missing values
print(df.isnull().sum())

# Remove any missing values if present (optional, depends on dataset)
df.dropna(inplace=True)

# Convert text messages to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
y = df['label']

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train and Evaluate Models
# Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluation
print("Naive Bayes Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_nb):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_nb):.4f}")

# Logistic Regression Classifier
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Evaluation
print("Logistic Regression Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.4f}")

# Support Vector Machine Classifier
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

# Evaluation
print("Support Vector Machine Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svc):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svc):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svc):.4f}")








