from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4215
)

model = make_pipeline(
    TfidfVectorizer(),
    LinearSVC()
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
# Appendix C.1: Testing a New Message
messages = [
    "You have won a free prize!",
    "Your class starts at 10 AM."
]

results = model.predict(messages)

for message, result in zip(messages, results):
    print(message, ":", result)


# Appendix C.2: Displaying a Graph
import matplotlib.pyplot as plt

df["label"].value_counts().plot(kind="bar")
plt.title("Spam and Legitimate Messages")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")
plt.show()


# Appendix C.3: Saving the Model
import joblib

joblib.dump(model, "spam_model.pkl")
    
