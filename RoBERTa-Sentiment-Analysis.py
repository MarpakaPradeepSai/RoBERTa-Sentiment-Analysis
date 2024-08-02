import streamlit as st
from transformers import pipeline

# Load the pre-trained sentiment analysis model (RoBERTa-based)
sentiment_analyzer = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

# Define a function to analyze sentiment
def roberta_analyze_sentiment(text):
    result = sentiment_analyzer(text)
    sentiment_label = result[0]['label']
    confidence = result[0]['score']

    # Map the model labels to more descriptive terms
    sentiment_mapping = {
        'LABEL_0': 'negative',  # assuming LABEL_0 is negative
        'LABEL_1': 'neutral',   # assuming LABEL_1 is neutral (if applicable)
        'LABEL_2': 'positive'   # assuming LABEL_2 is positive
    }

    sentiment = sentiment_mapping.get(sentiment_label, 'unknown')
    return sentiment, confidence

# Streamlit app
def main():
    st.title("Sentiment Analysis with RoBERTa")
    st.write("Enter a text below to analyze its sentiment.")

    # Text input
    user_input = st.text_area("Text")

    if st.button("Analyze"):
        if user_input:
            sentiment, confidence = roberta_analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
