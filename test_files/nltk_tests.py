from nltk.sentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    """Performs sentiment analysis on the given text and returns the sentiment scores."""
    # Create an instance of the SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    
    # Analyze sentiment
    sentiment_scores = sid.polarity_scores(text)
    
    return sentiment_scores

def classify_sentiment(sentiment_scores):
    """Classifies the sentiment of a given text.
    
    args:
        sentiment_scores (dict): The sentiment scores of the text.
        
    Returns:
        sentiment (string): The sentiment of the text."""
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Example usage
text = "I love this product! It's amazing."
sentiment_scores = analyze_sentiment(text)

# Extract sentiment metrics
positive_score = sentiment_scores['pos']
neutral_score = sentiment_scores['neu']
negative_score = sentiment_scores['neg']

print(f'Positive score: {positive_score}')
print(f'Neutral score: {neutral_score}')
print(f'Negative score: {negative_score}')

# Classify sentiment
print(classify_sentiment(sentiment_scores))