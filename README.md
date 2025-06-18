

Sentiment Analysis using NLP 🤖💬

Overview
A sentiment analysis project that uses natural language processing (NLP) to classify text as positive, negative, or neutral 🌟

Features
1. _Text Preprocessing_: Cleaning and preprocessing text data for analysis 🧹
2. _Sentiment Analysis_: Classifying text as positive, negative, or neutral using NLP models 🤖
3. _Model Evaluation_: Evaluating the performance of different NLP models 📊

Technologies Used
- Python 
- NLP libraries (e.g. NLTK, spaCy) 📚
- Machine learning libraries (e.g. scikit-learn) 🤖

Code Structure
- `data`: Directory containing text data for analysis 📁
- `models`: Directory containing NLP models for sentiment analysis 🤖
- `preprocessing`: Directory containing text preprocessing scripts 🧹

Example Use Cases
- Analyzing customer reviews for sentiment 📊
- Classifying text data for opinion mining 🤖
- Evaluating the sentiment of social media posts 📱

**Code Snippets**

*Importing Libraries*
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

*Preprocessing Text Data*
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace()).lower()
    # Split the text into words
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    return ' '.join(words)

*Training a Logistic Regression Classifier*
Create a CountVectorizer object
vectorizer = CountVectorizer()
Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
Train a Logistic Regression classifier with class weights
clf = LogisticRegression(max_iter=10000, class_weight='balanced')
clf.fit(X_train_count, y_train)

*Evaluating the Model*
Make predictions on the testing data
y_pred = clf.predict(X_test_count)
Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("The Classification Report:")
print(classification_report(y_test, y_pred))

*Model Metrics*
- *Accuracy*: Measures the proportion of correctly classified instances
- *Precision*: Measures the proportion of true positives among all predicted positive instances
- *Recall*: Measures the proportion of true positives among all actual positive instances
- *F1-score*: Harmonic mean of precision and recall

*Output*
[!https://drive.google.com/uc?id=1RSXgsOHNoLyV_7fIXnsqBaY42FqeI4iL](https://drive.google.com/file/d/1RSXgsOHNoLyV_7fIXnsqBaY42FqeI4iL/view?usp=drivesdk)

*Author*
- _Reaishma N_ 🙋‍♀️

*License*
MIT License 📄

