from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Step 1: Load and train model
df = pd.read_csv("emails.csv")
X = df['message']
y = df['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

# Step 2: Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    email_vectorized = vectorizer.transform([email_text])
    prediction = model.predict(email_vectorized)[0]
    return render_template('index.html', result=prediction, email=email_text)

if __name__ == '__main__':
    app.run(debug=True)
