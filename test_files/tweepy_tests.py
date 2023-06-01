import tweepy
    

def search_twitter(api_key, api_secret_key, access_token, access_token_secret, keywords, count=10):
    # Authenticate with Twitter API
    # auth = tweepy.AppAuthHandler(api_key, api_secret_key)
    client = tweepy.Client(
            consumer_key=api_key, 
            consumer_secret=api_secret_key, 
            access_token=access_token, 
            access_token_secret=access_token_secret)
    

    # Search for tweets
    tweets = client.search_recent_tweets(query=keywords, tweet_fields=["text"], max_results=count)

    # Process and print the tweets
    for tweet in tweets.data:
        print(f"{tweet.author.username}: {tweet.text}")
        print("------")

# Provide your Twitter API credentials


client = tweepy.Client(consumer_key= api_key,consumer_secret= api_secret_key,access_token= access_token,access_token_secret= access_token_secret)
query = 'news'
tweets = client.search_recent_tweets(query=query, max_results=10)
for tweet in tweets.data:
    print(tweet.text)