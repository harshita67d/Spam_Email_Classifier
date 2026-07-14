import pandas as pd
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv")

# Convert labels
data["label"] = data["label"].map({"ham":0,"spam":1})

# Clean text
def clean_text(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return text

data["text"] = data["text"].apply(clean_text)

# Split features and labels
X = data["text"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test model
prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

# Predict new email
email = ["Congratulations! You won a free laptop"]

email_vector = vectorizer.transform(email)

result = model.predict(email_vector)

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")
    