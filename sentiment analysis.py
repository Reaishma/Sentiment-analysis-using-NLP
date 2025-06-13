import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define the original dataframe
data = {
    "Text": ["I love this product!", "This movie is terrible.", "The food was okay.", 
             "The service was amazing!", "I'm so disappointed.", "The hotel was clean and comfortable.", 
             "The staff was unfriendly.", "The experience was average."],
    "Sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Positive", "Negative", "Neutral"]
}

df = pd.DataFrame(data)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join(e for e in text if e.isalnum() or e.isspace()).lower()
    
    # Split the text into words
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a string
    return ' '.join(words)

# Apply the preprocessing function to the text data
df['Text'] = df['Text'].apply(preprocess_text)

# Generate new sample data
new_data = {
    "Text": [
        "This product is amazing, I love it!",
        "The customer service was terrible.",
        "The food was okay, nothing special.",
        "I'm so happy with my purchase!",
        "The hotel room was clean and comfortable.",
        "The staff was unfriendly and unhelpful.",
        "The experience was average, nothing to write home about.",
        "I would definitely recommend this product!",
        "The quality was poor, not worth the price.",
        "The service was excellent, very responsive.",
        "The movie was boring and too long.",
        "I'm very satisfied with my purchase.",
        "The product was defective, not what I expected.",
        "The restaurant had great ambiance.",
        "The food was delicious, but the service was slow.",
        "I loved the new smartphone, it's so fast!",
        "The hotel breakfast was disappointing.",
        "The customer support team was very helpful.",
        "The product was not what I expected, it's too small.",
        "The restaurant had a great view.",
        "The service was slow, but the food was good.",
        "I'm so impressed with the new laptop, it's amazing!",
        "The hotel room was noisy, I couldn't sleep.",
        "The product was exactly what I needed, thanks!",
        "The customer service was unresponsive, not helpful."
    ],
    "Sentiment": [
        "Positive", "Negative", "Neutral", "Positive", "Positive",
        "Negative", "Neutral", "Positive", "Negative", "Positive",
        "Negative", "Positive", "Negative", "Positive", "Neutral",
        "Positive", "Negative", "Positive", "Negative", "Positive",
        "Neutral", "Positive", "Negative", "Positive", "Negative"
    ]
}

new_df = pd.DataFrame(new_data)

# Preprocess the new data
new_df['Text'] = new_df['Text'].apply(preprocess_text)

# Combine the new data with the existing data
combined_df = pd.concat([df, new_df], ignore_index=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_df['Text'], combined_df['Sentiment'], test_size=0.2, random_state=42)

# Create a CountVectorizer object
vectorizer = CountVectorizer()

# Fit the vectorizer to the training data and transform both the training and testing data
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Train a Logistic Regression classifier with class weights
clf = LogisticRegression(max_iter=10000, class_weight='balanced')
clf.fit(X_train_count, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_count)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("The Classification Report:")
print(classification_report(y_test, y_pred))