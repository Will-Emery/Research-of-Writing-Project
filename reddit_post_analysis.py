import praw
from reddit_auth import client_id, client_secret, user_agent
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import os


def search_reddit_for_phrase(phrase, subreddit_name, df):
    """Searches for a given phrase in a given subreddit and stores the results in a dataframe.
    
    ars:
        phrase (str): The phrase to search for.
        subreddit_name (str): The name of the subreddit to search in.
        df (pandas.DataFrame): The dataframe to store the results in.
        
    Returns:
        df (pandas.DataFrame): The dataframe with the results added."""
    # Create an instance of the Reddit API client
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )
    
    # Specify the subreddit you want to search in
    subreddit = reddit.subreddit(subreddit_name)
    
    # Search for the given phrase in the subreddit
    search_results = subreddit.search(phrase, time_filter = 'year', sort = 'top', limit=10)
    #NOTE: only pulling from the last year due to open release of chatGPT
    #NOTE: only pulling top 10 to try to keep representation of both sides equal. 
    # It was set to 100 but that resulted in mostly ai related posts being in the final dataset
    
    # Iterate over the search results
    for submission in search_results:
        #check to make sure the body contains the phrase we are looking for
        if phrase in submission.selftext:
            # Add the results to the dataframe
            #check to make sure the post isn't already in the dataframe
            if submission.id not in df['id'].values:
                #append in the format: 
                # 'title', 'url', 'body', 'score', 'id', 'author', 'num_comments', 'created', 'subreddit_class', 'sentiment'
                submission_sentiment = analyze_sentiment(submission.selftext)
                df.loc[len(df.index)] = [submission.title, submission.url, submission.selftext, 
                                         submission.score, submission.id, submission.author, 
                                         submission.num_comments, submission.created, ' ', submission_sentiment]
                
                # Print the title and URL of each included submission
                print(f'Title: {submission.title}')
                print(f'URL: {submission.url}')
                print(f'submission sentiment: {submission_sentiment}')
                print('---')
    return df


def handel_subreddit_list_writing_related(subreddit_list):
    """Handles the list of subreddits to search through.
    
    args:
        subreddit_list (list): The list of subreddits to search through.
        
    Returns:
        df (pandas.DataFrame): The dataframe with the results added."""
    
    subreddit_classifcation = 'writing'
    df = make_dataframe()

    print(df)
    
    for subreddit in subreddit_list:
        df = search_reddit_for_phrase('ChatGPT', subreddit, df)
        df = search_reddit_for_phrase('chatGPT', subreddit, df)
        df = search_reddit_for_phrase(' AI ', subreddit, df)
        df = search_reddit_for_phrase(' ai ', subreddit, df)
        df = search_reddit_for_phrase('Large Language Models', subreddit, df)
        df = search_reddit_for_phrase('large language models', subreddit, df)
        df = search_reddit_for_phrase('llm', subreddit, df)
        df = search_reddit_for_phrase('LLM', subreddit, df)

    #fill in the subreddit classification column
    df['subreddit_class'] = subreddit_classifcation
    return df


def handle_subbreddit_list_ai_related(subreddit_list):
    """Handles the list of subreddits to search through.
    
    args:
        subreddit_list (list): The list of subreddits to search through.
        
    Returns:
        df (pandas.DataFrame): The dataframe with the results added."""
    
    subreddit_classifcation = 'ai'
    df = make_dataframe()
    
    for subreddit in subreddit_list:
        df = search_reddit_for_phrase('writing', subreddit, df)
        df = search_reddit_for_phrase('Writing', subreddit, df)
        df = search_reddit_for_phrase('writer', subreddit, df)
        df = search_reddit_for_phrase('Writer', subreddit, df)
        df = search_reddit_for_phrase('author', subreddit, df)
        df = search_reddit_for_phrase('Author', subreddit, df)
        df = search_reddit_for_phrase('book', subreddit, df)
        df = search_reddit_for_phrase('Book', subreddit, df)
        df = search_reddit_for_phrase('novel', subreddit, df)
        df = search_reddit_for_phrase('Novel', subreddit, df)
    
    #fill in the subreddit classification column
    df['subreddit_class'] = subreddit_classifcation
    return df


def analyze_sentiment(text):
    """Analyzes the sentiment of a given text.
    
    args:
        text (str): The text to analyze.
        
    Returns:
        sentiment (string): The sentiment of the text."""
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    classified_sentiment = classify_sentiment(sentiment_scores)

    return classified_sentiment


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
    

def get_length_of_body(df):
    """Gets the length of the body of each post in the dataframe.
    
    args:
        df (pandas.DataFrame): The dataframe to get the length of the body of each post from.
        
    Returns:
        df (pandas.DataFrame): The dataframe with the length of the body of each post added."""
    df['body_length'] = df['body'].str.len()
    return df


def pull_highest_score_posts(df):
    """Pulls the highest scoring posts of different categories.
    
    args:
        df (pandas.DataFrame): The dataframe to pull the highest scoring posts from.
        
    Returns:
        a df of the highest scoring posts of different categories."""
    
    return_df = pd.DataFrame(columns=['title', 'url', 'body', 'score', 'id', 'author', 'num_comments', 'created', 'subreddit_class', 'sentiment', 'reason'])
    highest_score_list = []
    #get the highest scoring post
    highest_scoring_post = df.iloc[0]
    highest_scoring_post['reason'] = 'highest scoring post'
    return_df[len(return_df.index)] = highest_scoring_post
    highest_score_list.append(highest_scoring_post)
    #get the highest scoring post with a positive sentiment
    highest_scoring_positive_post = df[df['sentiment'] == 'positive'].iloc[0]
    highest_scoring_positive_post['reason'] = 'highest scoring positive post'
    return_df[len(return_df.index)] = highest_scoring_positive_post
    highest_score_list.append(highest_scoring_positive_post)
    #get the highest scoring post with a negative sentiment
    highest_scoring_negative_post = df[df['sentiment'] == 'negative'].iloc[0]
    highest_scoring_negative_post['reason'] = 'highest scoring negative post'
    return_df[len(return_df.index)] = highest_scoring_negative_post
    highest_score_list.append(highest_scoring_negative_post)
    #get the highest scoring post with a neutral sentiment
    highest_scoring_neutral_post = df[df['sentiment'] == 'neutral'].iloc[0]
    highest_scoring_neutral_post['reason'] = 'highest scoring neutral post'
    return_df[len(return_df.index)] = highest_scoring_neutral_post
    highest_score_list.append(highest_scoring_neutral_post)

    #get the highest scoring post with a writing subreddit classification
    highest_scoring_writing_post = df[df['subreddit_class'] == 'writing'].iloc[0]
    highest_scoring_writing_post['reason'] = 'highest scoring writing post'
    return_df[len(return_df.index)] = highest_scoring_writing_post
    highest_score_list.append(highest_scoring_writing_post)
    #get the highest scoring post with a writing subreddit classification and a positive sentiment
    highest_scoring_writing_positive_post = df[(df['subreddit_class'] == 'writing') & (df['sentiment'] == 'positive')].iloc[0]
    highest_scoring_writing_positive_post['reason'] = 'highest scoring writing positive post'
    return_df[len(return_df.index)] = highest_scoring_writing_positive_post
    highest_score_list.append(highest_scoring_writing_positive_post)
    #get the highest scoring post with a writing subreddit classification and a negative sentiment
    highest_scoring_writing_negative_post = df[(df['subreddit_class'] == 'writing') & (df['sentiment'] == 'negative')].iloc[0]
    highest_scoring_writing_negative_post['reason'] = 'highest scoring writing negative post'
    return_df[len(return_df.index)] = highest_scoring_writing_negative_post
    highest_score_list.append(highest_scoring_writing_negative_post)
    #get the highest scoring post with a writing subreddit classification and a neutral sentiment
    highest_scoring_writing_neutral_post = df[(df['subreddit_class'] == 'writing') & (df['sentiment'] == 'neutral')].iloc[0]
    highest_scoring_writing_neutral_post['reason'] = 'highest scoring writing neutral post'
    return_df[len(return_df.index)] = highest_scoring_writing_neutral_post
    highest_score_list.append(highest_scoring_writing_neutral_post)

    #get the highest scoring post with an ai subreddit classification
    highest_scoring_ai_post = df[df['subreddit_class'] == 'ai'].iloc[0] #highest here will be the same as the highest overall post
    highest_scoring_ai_post['reason'] = 'highest scoring ai post'
    return_df[len(return_df.index)] = highest_scoring_ai_post
    highest_score_list.append(highest_scoring_ai_post)
    #get the highest scoring post with an ai subreddit classification and a positive sentiment
    highest_scoring_ai_positive_post = df[(df['subreddit_class'] == 'ai') & (df['sentiment'] == 'positive')].iloc[0]
    highest_scoring_ai_positive_post['reason'] = 'highest scoring ai positive post'
    return_df[len(return_df.index)] = highest_scoring_ai_positive_post
    highest_score_list.append(highest_scoring_ai_positive_post)
    #get the highest scoring post with an ai subreddit classification and a negative sentiment
    highest_scoring_ai_negative_post = df[(df['subreddit_class'] == 'ai') & (df['sentiment'] == 'negative')].iloc[0]
    highest_scoring_ai_negative_post['reason'] = 'highest scoring ai negative post'
    return_df[len(return_df.index)] = highest_scoring_ai_negative_post
    highest_score_list.append(highest_scoring_ai_negative_post)
    #get the highest scoring post with an ai subreddit classification and a neutral sentiment
    highest_scoring_ai_neutral_post = df[(df['subreddit_class'] == 'ai') & (df['sentiment'] == 'neutral')].iloc[0]
    highest_scoring_ai_neutral_post['reason'] = 'highest scoring ai neutral post'
    return_df[len(return_df.index)] = highest_scoring_ai_neutral_post
    highest_score_list.append(highest_scoring_ai_neutral_post)

    for post in highest_score_list:
        print(post)
        print('---')

    #pull the links out of the highest scoring posts
    for post in highest_score_list:
        append_to_file('highest_scoring_posts.txt', 'reason: '+ post['reason'])
        append_to_file('highest_scoring_posts.txt', 'url: ' + post['url'])
        append_to_file('highest_scoring_posts.txt', 'title : ' + post['title'])
        append_to_file('highest_scoring_posts.txt', 'sentiment: ' + post['sentiment'])
        append_to_file('highest_scoring_posts.txt', 'body: ' + post['body'])
        append_to_file('highest_scoring_posts.txt', '---')

    #NOTE: the df is messed up. disregard
    return return_df
    

"""
_____________________
Graphing functions
_____________________
"""
def graph_score_to_sentiment(df):
    """Graphs the score to sentiment ratio.
    
    args:
        df (pandas.DataFrame): The dataframe to graph."""
    
    #find the average score for each sentiment
    df_positive = df[df['sentiment'] == 'positive']
    df_negative = df[df['sentiment'] == 'negative']
    df_neutral = df[df['sentiment'] == 'neutral']

    #find the average score for each sentiment
    average_positive_score = df_positive['score'].mean()
    average_negative_score = df_negative['score'].mean()
    average_neutral_score = df_neutral['score'].mean()

    #plot the data in a bar chart
    plt.bar(['positive', 'negative', 'neutral'], [average_positive_score, average_negative_score, average_neutral_score])
    plt.title('Sentiment and Score For All subreddits')
    for i, v in enumerate([average_positive_score, average_negative_score, average_neutral_score]):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')    
    plt.show()

    #now do the same thing but for only writing related subreddits
    df_writing = df[df['subreddit_class'] == 'writing']
    df_writing_positive = df_writing[df_writing['sentiment'] == 'positive']
    df_writing_negative = df_writing[df_writing['sentiment'] == 'negative']
    df_writing_neutral = df_writing[df_writing['sentiment'] == 'neutral']
    
    #find the average score for each sentiment
    average_writing_positive_score = df_writing_positive['score'].mean()
    average_writing_negative_score = df_writing_negative['score'].mean()
    average_writing_neutral_score = df_writing_neutral['score'].mean()

    #plot the data in a bar chart
    plt.bar(['positive', 'negative', 'neutral'], [average_writing_positive_score, average_writing_negative_score, average_writing_neutral_score])
    plt.title('Sentiment and Score For Writing subreddits')
    for i, v in enumerate([average_writing_positive_score, average_writing_negative_score, average_writing_neutral_score]):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.show()

    #now do the same thing but for only ai related subreddits
    df_ai = df[df['subreddit_class'] == 'ai']
    df_ai_positive = df_ai[df_ai['sentiment'] == 'positive']
    df_ai_negative = df_ai[df_ai['sentiment'] == 'negative']
    df_ai_neutral = df_ai[df_ai['sentiment'] == 'neutral']
    
    #find the average score for each sentiment
    average_ai_positive_score = df_ai_positive['score'].mean()
    average_ai_negative_score = df_ai_negative['score'].mean()
    average_ai_neutral_score = df_ai_neutral['score'].mean()

    #plot the data in a bar chart
    plt.bar(['positive', 'negative', 'neutral'], [average_ai_positive_score, average_ai_negative_score, average_ai_neutral_score])
    plt.title('Sentiment and Score For AI related subreddits')
    for i, v in enumerate([average_ai_positive_score, average_ai_negative_score, average_ai_neutral_score]):
        plt.annotate(str(v), xy=(i, v), ha='center', va='bottom')
    plt.show()


def graph_post_distribution(df):
    """Graphs the distribution of posts.
    
    args:
        df (pandas.DataFrame): The dataframe to graph."""
    
    #plot the data in a pie chart
    df['subreddit_class'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Post Distribution For All subreddits')
    plt.show()


def graph_sentiment_distribution(df):
    """Graphs the distribution of sentiment.
    
    args:
        df (pandas.DataFrame): The dataframe to graph."""
    
    #plot the data in a pie chart
    df['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sentiment Distribution For All subreddits')
    plt.show()

    #now graph the data for only writing related subreddits
    df_writing = df[df['subreddit_class'] == 'writing']
    df_writing['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sentiment Distribution For Writing related subreddits')
    plt.show()

    #now graph the data for only ai related subreddits
    df_ai = df[df['subreddit_class'] == 'ai']
    df_ai['sentiment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Sentiment Distribution For AI related subreddits')
    plt.show()


def graph_score_and_comments(df):
    """Graphs score and comments of the posts.
    
    args:
        df (pandas.DataFrame): The dataframe to graph."""
    
    #plot the data on a scattler plot
    plt.scatter(df['score'], df['num_comments'])
    plt.xlabel('Score')
    plt.ylabel('Number of Comments')
    plt.title('Score vs Number of Comments')
    plt.show()    


def graph_sentiment_and_length(df):
    """Graphs sentiment and length of the posts.
    
    args:
        df (pandas.DataFrame): The dataframe to graph."""
    
    #plot the data on a scattler plot
    plt.scatter(df['score'], df['body_length'])
    plt.xlabel('Sentiment')
    plt.ylabel('Length of Body')
    plt.title('Score vs Length of Body')
    plt.show()


"""
_____________________
Helper Functions
_____________________
"""

def make_dataframe():
    """Creates a dataframe with the columns we want to store from the Reddit API."""
    df = pd.DataFrame(columns=['title', 'url', 'body', 'score', 'id', 'author', 
                                 'num_comments', 'created', 'subreddit_class', 'sentiment'])
    return df


def trim_empty_lines_from_csv(file_name):
    """Trims partially empty lines from a csv file which result from newline and
    other weird characters in raw data."""
    with open(file_name, 'r') as f:
        lines = f.readlines()
    with open(file_name, 'w') as f:
        for line in lines:
            if line.strip():
                f.write(line)


def write_iterable_to_file(file_name, iterable):
    """Writes an iterable to a file.
    
    args:
        file_name (str): The name of the file to write to.
        iterable (iterable): The iterable to write to the file."""
    with open(file_name, 'w') as f:
        for item in iterable:
            f.write(str(item))
            f.write('----\n')

def append_to_file(file_name, text):
    """Appends text to a file.
    
    args:
        file_name (str): The name of the file to append to.
        text (str): The text to append to the file."""
    with open(file_name, 'a') as f:
        f.write(text)
        f.write('\n')



if __name__ == '__main__':
    results_df = make_dataframe()

    writing_subreddit_list = ['writing', 'books', 'nanowrimo', 'writerchat',
                              'writingprompts', 'freelancewriters', 'fantasywriters', 
                              'scifiwriting', 'pubtips', 'selfpublish']
    ai_subreddit_list = ['artificial', 'openai', 'MachineLearning', 'ChatGPT', 'GPT3',
                         'singlarity']

    results_df = handel_subreddit_list_writing_related(writing_subreddit_list)

    #add the ai related subreddits
    results_df = results_df._append(handle_subbreddit_list_ai_related(ai_subreddit_list), ignore_index=True)

    results_df.to_csv('reddit_results_raw.csv')
    print("raw results saved")
    print(results_df)

    #sort the results by score
    results_df = results_df.sort_values(by=['score'], ascending=False)
    print("results sorted by score")
    print(str(len(results_df)) + " posts found")

    #get the highest scoring posts
    highest_scoreing_posts = pull_highest_score_posts(results_df)
    #highest_scoreing_posts.to_csv('reddit_results_highest_scoreing_posts.csv')
    #NOTE: the df and csv file are incredibly messed up. disregard
    print("highest scoring posts saved")

    #get the length of the body of each post
    results_df = get_length_of_body(results_df)
    results_df.to_csv('reddit_results_sorted.csv')
    trim_empty_lines_from_csv('reddit_results_sorted.csv')
    print("empty lines trimmed")


    #graphing functions
    graph_score_to_sentiment(results_df)
    graph_post_distribution(results_df)
    graph_sentiment_distribution(results_df)
    graph_score_and_comments(results_df)
    graph_sentiment_and_length(results_df)