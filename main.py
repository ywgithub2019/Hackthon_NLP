import io
import keras
import spacy
import pickle
import re
import pandas as pd
import numpy as np
from classifier import *
from keras.models import load_model
from keras.layers import Add
from keras.utils import to_categorical
import keras.backend as K
from keras.preprocessing.text import one_hot
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense, Activation , Conv1D, MaxPooling1D, Bidirectional, Reshape, Flatten
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Model
from keras.layers import Concatenate, Input, Dense
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm




#### Data cleaning Functions ###

# As we want to do the data cleaning for every datasets (training & test),
# we will do data cleaning functions that we will put both in the train and the predict methods

def aspect_term(df, colnum):
    """Only lower cases in the tweets
    """

    
    for col_index in colnum:
        lower_words = []
        for index, row in df.iterrows():
            lower_words.append(row[col_index].lower())
        df[col_index] = lower_words
    


    # To make the best out of our data, we will also study the entire sentences. 
    # Thus, we will clean it:

def sentence_modifications(df, colnum):
    """Do some modifcations on the tweets 
    """
    for col_index in colnum:
        lower_reviews = []
        # We only want lower cases
        for index, row in df.iterrows():
            lower_reviews.append(row[col_index].lower())
            row[col_index] = re.sub("[!@#$+%*:()'-]", ' ', row[col_index])
        df[col_index] = lower_reviews
        
        

        # We use spacy 
        nlp = spacy.load('en')
        cleaned_reviews = []
        
        # We want to remove stop words and punctuation:
        for doc in nlp.pipe(df[col_index].astype('unicode').values):
            if doc.is_parsed:
                cleaned_reviews.append(' '.join([tok.lemma_ for tok in doc if (not tok.is_stop and not tok.is_punct)])) 
            
            else:
                # (We don't want an element of the list to be empty)
                cleaned_reviews.append('') 
            
        df[col_index] = cleaned_reviews


    print("Tweets modification done ")


######### End of data cleaning functions ##########


# Build the sentiment words table
    
    
# Here, we just convert string to 1, -1 and 0 for two reviews tables:
    
new_path = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/word_sentiment.xlsx'
word_sentiment = pd.read_excel(new_path)
word_sentiment["Positive"].loc[word_sentiment.Positive == "Positive"] = 1
word_sentiment["Positive"].loc[word_sentiment.Positive == "Negative"] = -1 
 
new_path = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/restaurantreviews.csv'
word_sentiment2 = pd.read_csv(new_path, sep='\t', header= None, encoding = "ISO-8859-1")
word_sentiment2[0].loc[word_sentiment2[0] == "positive"] = 1
word_sentiment2[0].loc[word_sentiment2[0] == "negative"] = -1 
word_sentiment2[0].loc[word_sentiment2[0] == "neutral"] = 0 

# For the second table, just two columns are interesting for us
word_sentiment2 = pd.concat([word_sentiment2[0], word_sentiment2[4]], axis = 1)

# We want to clean some reviews of the second table
aspect_term(word_sentiment2, [4])
sentence_modifications(word_sentiment2, [4])


# Let's now merge the two dataframes:

columnsTitles=[4,0]
word_sentiment2 = word_sentiment2.reindex(columns=columnsTitles)
word_sentiment2 = word_sentiment2.rename(columns = {0 : 1, 4: 0})
word_sentiment = word_sentiment.rename(columns= {"a+": 0, "Positive": 1})



training_reviews = pd.concat([word_sentiment, word_sentiment2])



######## Tweets part ##############


trainfile = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/tweets_dataset.csv'
    
        
# First, we will clean the training set:

tweets_set = pd.read_csv(trainfile, sep=',', header= None, encoding = "ISO-8859-1")
tweets_set = tweets_set.drop(0, axis=0)
tweets_set = tweets_set.reset_index()
# Do the cleaning: Lower cases, no stop word, no punctuation:
columns_index = [i for i in range(1,11)]
aspect_term(tweets_set, columns_index)
sentence_modifications(tweets_set, columns_index)

        
# First, we create a tokenizer:
voc_size = 1000
tokenizer = Tokenizer(num_words = voc_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)

# Give the list of texts to train our tokenizer
text_for_tokenizer = pd.concat([tweets_set[1], tweets_set[2], tweets_set[3], tweets_set[4], tweets_set[5], tweets_set[6], tweets_set[7], tweets_set[8], tweets_set[9], tweets_set[10]])
fit_tokenizer_1 = pd.concat([word_sentiment[0], word_sentiment2[0]])
text_for_tokenizer = pd.concat([text_for_tokenizer, fit_tokenizer_1])
tokenizer.fit_on_texts(text_for_tokenizer)
        


### Let's transform reviews into sentiment score (-1, 0, 1)
final_dataset = tweets_set   
classification_tweets(tokenizer,training_reviews,final_dataset)

for i in range(1,11):
    final_dataset[i] = final_dataset[i] * final_dataset[i+10]
final_dataset = pd.concat([final_dataset[1],final_dataset[2],
                           final_dataset[3],
                           final_dataset[4],
                           final_dataset[5],
                           final_dataset[6],
                           final_dataset[7],
                           final_dataset[8],
                           final_dataset[9],
                           final_dataset[10]],axis=1)
######### Financial informations ############        
        
new_path = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/financial_infos.xlsx'
financial_dataset = pd.read_excel(new_path)
financial_dataset = financial_dataset.drop("date", axis=1)
new_path = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/googletrend.csv'
googletrend = pd.read_csv(new_path, sep=',', header= None)
googletrend = googletrend.drop(0, axis=0)
googletrend = googletrend.reset_index()
# Based on what we found out, we will only keep some of the columns:
financial_dataset = pd.concat([financial_dataset["BCHAIN_HashRate"],
                               financial_dataset["LBMA_GOLD_USD_AM"],
                               googletrend.loc[:,2]],axis=1)
        
#########################################

# Now we build the final dataset. In order to do that:
# 1) Build a sentiment analysis model 
# 2) Convert all our sentences in positive (1) or negative (-1)
final_dataset = pd.concat([final_dataset, financial_dataset], axis=1)




        
##############################################
# The next part is to classify each tweet. 
# We will use sentiment analysis. The training set will be this one: 






        
        

