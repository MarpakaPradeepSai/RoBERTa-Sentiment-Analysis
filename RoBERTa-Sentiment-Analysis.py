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

    # HTML for the styled button
    button_html = """
    <style>
    .big-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    </style>
    <a class="big-button" href="javascript:void(0);" onclick="document.getElementById('analyze-button').click()">Analyze</a>
    """

    # Display the styled button
    st.markdown(button_html, unsafe_allow_html=True)
    
    # Hidden button to handle the click event
    analyze_button = st.button("Analyze", key='analyze-button')

    if analyze_button:
        if user_input:
            sentiment, confidence = roberta_analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
