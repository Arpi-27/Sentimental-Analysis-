!pip install streamlit pyngrok pandas numpy scikit-learn nltk -q
%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

st.title("🧠 Sentiment Analysis Web App")
st.write("This app analyzes the sentiment (Positive / Negative / Neutral) of your input text or review.")

uploaded_file = st.file_uploader("Upload your dataset (CSV with 'review' and 'sentiment' columns):", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    stop_words = stopwords.words('english')

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text))
        text = text.lower().split()
        text = [word for word in text if word not in stop_words]
        return " ".join(text)

    data['clean_text'] = data['review'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['clean_text'], data['sentiment'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer()
    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    st.write(f"**Model Accuracy:** {acc*100:.2f}%")

    st.subheader("Test the model with your own text")
    user_input = st.text_area("Enter your text or review:")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            if prediction.lower() == "positive":
                st.success(f"✅ Sentiment: {prediction}")
            elif prediction.lower() == "negative":
                st.error(f"❌ Sentiment: {prediction}")
            else:
                st.info(f"😐 Sentiment: {prediction}")
else:
    st.info("👆 Please upload your dataset to start.")

from pyngrok import ngrok
import threading
import time
import os



ngrok.set_auth_token("31BzCoEu8sUEkj5ch2j7o4SACWj_5Bp9oH5gu6AebK4hZ9Qt8")

#Streamlit server
def run_app():
    os.system("streamlit run app.py --server.port 8501")

thread = threading.Thread(target=run_app)
thread.start()

time.sleep(5)

# Create public tunnel
public_url = ngrok.connect(addr="8501")
print("Your app is live here:", public_url)
