import tweepy 
from twitter_authentication import consumer_key, consumer_secret, access_token, access_token_secret, bearer_token
import pandas as pd


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth)

dataset = []
for tweet in tweepy.Cursor(api.search_tweets, q='iphone 14').items(10):

    tweet_data = {'user_name':tweet.user.screen_name,
                  'text':tweet.text
                 }

    dataset.append(tweet_data)

df = pd.DataFrame(dataset)
print(df)