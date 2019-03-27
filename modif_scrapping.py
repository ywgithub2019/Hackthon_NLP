import pandas as pd 



# Import data
tweets = pd.read_csv('/.../twitter.csv', sep=',')



# Only keep the day and the year of the tweet
tweets["published_date"] = tweets["published_date"].str.split('- ').str[1]

# Sort values per retweets
tweets = tweets.sort_values(by=['retweets'], ascending = False)

# Then we keep the first n rows per day, and we reset the index of the dataframe 
tweets = tweets.groupby('published_date', as_index=False).head(3)
tweets = tweets.reset_index(drop=True)

# Finally, we just fill the nan values 
tweets = tweets.fillna(0)




