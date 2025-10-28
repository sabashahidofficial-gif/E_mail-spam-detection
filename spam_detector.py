import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#step 1: Load data
df = pd.read_csv("emails.csv")
print("✔ Dataset loaded successfully!")
print(df.head())
#step 2: Split data
X = df['message'] # email text
y = df['label'] # spam or ham
# step 3: Convert text to numbers(Bag of words)
Vectorizer = CountVectorizer()
X_vectorizer = Vectorizer.fit_transform(X)
# step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorizer, y, test_size=0.2)
# step 5: Train model
model = MultinomialNB()
model.fit(X_train, y_train)
# step 6: Test model
y_pred = model.predict (X_test)
print("\n ✔ Model trained successfully!")
print("📈 Accuracy:", accuracy_score(y_test, y_pred))
print("\n 📜 Classification report:\n", classification_report(y_test, y_pred))
# step 7: Test with custom input test_emails # step 7: Test with custom input
test_emails = [
    "congratulations! you have won a free laptop. Click to claim.",
    "Hey are we still on for the meeting?",
    "Claim your $1000 reward now!"
]
test_vectorized = Vectorizer.transform(test_emails)
predictions = model.predict(test_vectorized)
print("\n 📩 custom Email prediction:")
for email, label in zip(test_emails, predictions):
    print(f" '{email}'➡{label}")
