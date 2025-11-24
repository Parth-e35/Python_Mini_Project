# spam_detector.py

import streamlit as st
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Email Spam Detector", layout="centered")

# ------------------------------------------
# 1. Sample Data to Train the Model (Built-in)
# ------------------------------------------
training_emails = [
    "Congratulations! You won a lottery of $10000. Claim your prize now.",
    "Free vacation waiting for you. Click to claim.",
    "Lowest price medicines available online.",
    "Hello brother, how are you?",
    "Meeting scheduled for tomorrow at 5 PM.",
    "Don't forget our project submission.",
]

training_labels = ["spam", "spam", "spam", "ham", "ham", "ham"]

# Vectorizer + Naive Bayes Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_emails)
model = MultinomialNB()
model.fit(X, training_labels)

# ------------------------------------------
# 2. Text Cleaning Function
# ------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------------------
# 3. Streamlit UI - User Input Box
# ------------------------------------------
st.title("📧 Email Spam Detection App")
st.write("Enter any email/message below and check if it is **Spam** or **Not Spam**.")

st.markdown("---")

user_email = st.text_area("✍️ Type your email/message here:", height=180)

# ------------------------------------------
# 4. Predict Button
# ------------------------------------------
if st.button("🔍 Predict"):
    if user_email.strip() == "":
        st.warning("⚠ Please enter some text to predict.")
    else:
        cleaned_email = clean_text(user_email)
        transformed = vectorizer.transform([cleaned_email])
        prediction = model.predict(transformed)[0]

        if prediction == "spam":
            st.error("🚨 The message is **SPAM**.")
        else:
            st.success("✔ The message is **NOT SPAM (HAM)**.")

st.markdown("---")
st.caption("Mini Project | Simple Spam Detection using Naive Bayes")