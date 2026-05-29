
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import spacy

# Ensure NLTK data is downloaded for deployment
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load the spaCy model once
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# Load the TF-IDF vectorizer and the trained model
@st.cache_resource
def load_artifacts():
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('sentiment_model.pkl')
    return tfidf_vectorizer, model

tfidf_vectorizer, model = load_artifacts()

# Define the custom stopwords (must match the training preprocessing)
custom_stopwords = set(stopwords.words('english'))
custom_stopwords = custom_stopwords - {"not", "no", "never", "nor"}

# Text cleaning function (must match the training preprocessing)
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'http\\S+|www\\S+', '', text)  # Remove links
    text = re.sub(r'\\d+', '', text)  # Remove numbers

    # SpaCy processing
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in custom_stopwords
        and not token.is_punct
        and token.is_alpha
    ]
    return " ".join(tokens)

st.title("AI/ML Based Sentiment Analysis System")
st.write("Analyze text sentiment using NLP and Machine Learning.")
st.write("Developed by Sourabh Kallole")

# Text input from user
user_input = st.text_area("Enter your review here:", "")

if st.button("Analyze Sentiment"):
    if user_input:
        # Clean the input text
        cleaned_input = clean_text(user_input)

        # Transform the cleaned text using the loaded TF-IDF vectorizer
        input_vectorized = tfidf_vectorizer.transform([cleaned_input])

        # Make a prediction
        prediction = model.predict(input_vectorized)
        predicted_sentiment = prediction[0]

        st.write(f"Predicted Sentiment: **{predicted_sentiment}**")

    else:
        st.warning("Please enter some text to analyze.")
