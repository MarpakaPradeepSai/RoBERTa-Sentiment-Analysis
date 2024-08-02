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
        'LABEL_0': 'Negative😡',  # assuming LABEL_0 is negative
        'LABEL_1': 'Neutral😐',   # assuming LABEL_1 is neutral (if applicable)
        'LABEL_2': 'Positive😊'   # assuming LABEL_2 is positive
    }

    sentiment = sentiment_mapping.get(sentiment_label, 'unknown')
    return sentiment, confidence

# Streamlit app
def main():
    st.title("Sentiment Analysis with RoBERTa")
    
    # Custom styling for the instruction text
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            border: 1px solid #d3d3d3;
            padding: 10px;
            font-size: 16px;
        }
        .stTextArea textarea:focus {
            border: 1px solid blue;
            outline: none;
        }
        .stButton > button {
            background-color: green;
            color: white !important;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
        .stButton > button:focus {
            background-color: green;
            color: white !important;
        }
        .stButton > button:hover {
            background-color: darkgreen;
            color: white !important;
        }
        .custom-text {
            font-size: 20px;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<p class="custom-text">Enter a text below to analyze its sentiment.</p>', unsafe_allow_html=True)

    # Text input
    user_input = st.text_area("Text")

    if st.button("Analyze"):
        if user_input:
            sentiment, confidence = roberta_analyze_sentiment(user_input)
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Confidence:** {confidence:.2f}")
        else:
            st.write("Please enter some text.")

if __name__ == "__main__":
    main()
