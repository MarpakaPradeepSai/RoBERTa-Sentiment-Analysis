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
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

# Inject custom CSS for button styling
st.markdown(
    """
    <style>
    .custom-button {
        background-color: #28a745; /* Green color */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }

    .custom-button:hover {
        background-color: #218838; /* Darker green for hover effect */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a custom button using HTML and JavaScript
if st.markdown('<button class="custom-button" id="analyze-button">Analyze Sentiment</button>', unsafe_allow_html=True):
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, confidence = roberta_analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("Please enter some text.")
