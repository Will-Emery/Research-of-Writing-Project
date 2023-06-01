import tweepy
from twitter_authentication import consumer_key, consumer_secret, access_token, access_token_secret, bearer_token

def search_tweets(keyword):
    # Authenticate to Twitter
    # Create a client object
    client = tweepy.Client(bearer_token=bearer_token,
                           consumer_key=consumer_key,
                           consumer_secret=consumer_secret,
                           access_token=access_token,
                           access_token_secret=access_token_secret)

    # Search for tweets that contain the keyword
    tweets = client.search_recent_tweets(query=keyword, lang="en")

    print(f"Found {len(tweets)} tweets containing the keyword {keyword}.")

    # Print the tweets
    for tweet in tweets:
        print(tweet.text)


def post_tweets(tweet_text):
    #Authenticate to Twitter
    
    # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # auth.set_access_token(access_token, access_token_secret)

    # Create a client object
    client = tweepy.Client(
        consumer_key=consumer_key,
        consumer_secret=consumer_secret,
        access_token=access_token,
        access_token_secret=access_token_secret
    )

    # Post a tweet
    client.create_tweet(text=tweet_text)
    print(f"Tweet posted: {tweet_text}")


if __name__ == "__main__":
    start = input("Would you like to search or post a tweet?")
    while start != "search" and start != "post":
        start = input("Please enter either 'search' or 'post': ")
    
    if start == "search":
        keyword = input("Enter a keyword to search for: ")
        search_tweets(keyword)
    else :
        tweet_text = input("Enter the text of the tweet to post: ")
        post_tweets(tweet_text)