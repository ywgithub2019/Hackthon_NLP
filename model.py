import io
import keras
import spacy
import pickle
import re
import pandas as pd
import numpy as np
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

tokenizer.fit_on_texts(text_for_tokenizer)
        
# Then, we save the existing tokenizer to apply it on new data.
with open('tokenizer_file', 'wb') as handle:
    pickle.dump(tokenizer, handle)

new_dataframe = []      

new_dataframe = pd.concat([pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[1])),
           pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[2])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[3])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[4])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[5])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[6])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[7])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[8])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[9])),
            pd.DataFrame(tokenizer.texts_to_matrix(tweets_set[10])),
            tweets_set[11],tweets_set[12],tweets_set[13],tweets_set[14],
            tweets_set[15],tweets_set[16],tweets_set[17],tweets_set[18],
            tweets_set[19],tweets_set[20]], axis=1)  
        

########## End of tweets part #################


######### Financial informations ############        
        
new_path = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/financial_infos.xlsx'
financial_dataset = pd.read_excel(new_path)
financial_dataset = financial_dataset.drop("date", axis=1)

        
        
##############################################

# Now we build the final dataset:
final_dataset = pd.concat([new_dataframe, financial_dataset], axis=1)
            


# And upload the labels 
labelsfile = '/Users/nolwenbrosson/Desktop/Cours Nolwen/Cours Centrale/Hackaton/data/label_bitcoins.csv'
labels = pd.read_csv(labelsfile, sep='\t', header= None, encoding = "ISO-8859-1")
labels = labels[1309:1674] 
labels = labels.drop(0, axis=1)
labels = labels.drop(1, axis=1)


polarity_encoder = LabelEncoder()
transform_polarity = polarity_encoder.fit_transform(labels)
onehot_polarity = pd.DataFrame(to_categorical(transform_polarity))
labels = onehot_polarity


#### Let's try a simple deep learning model now #####
from keras.optimizers import SGD
model = Sequential()
# Input of size * x 14012 and output of size 512
model.add(Dense(1024, input_shape=(10023,)))
# Relu activation function
model.add(Activation('relu'))
#model.add(Dense(512))
#model.add(Activation('relu'))
# Final output of size 3 (three posible polarities)
model.add(Dense(2))
model.add(Activation('softmax'))
opt = SGD(lr=0.0001,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(final_dataset, labels, epochs=10, verbose=1)
model.save('model.simple') # 
        
  
        
        

