
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('stopwords')

# Load the CSV files into pandas DataFrames
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Add a label column to each DataFrame
fake_df['label'] = 0  # 0 for fake news
true_df['label'] = 1  # 1 for true news

# Combine the two DataFrames
data = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

data['text'] = data['text'].apply(preprocess_text)
X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# You can add evaluation here if you want
y_pred = model.predict(X_test_vectorized)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
