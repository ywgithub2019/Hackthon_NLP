import io
import keras
import spacy
import pickle
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


def classification_tweets(tokenizer,training_reviews,final_dataset):
    print("==> Classification model Starting... ")  
    # Transform the reviews and the aspect terms to matrix of size len(train_set[4] / voc_size)
    training_matrix = tokenizer.texts_to_matrix(training_reviews[0])
    labels = training_reviews[1]
    
    clf = svm.SVC(gamma='scale')
    clf.fit(training_matrix, labels)
    print("==> Training Done...")
    for i in range(1,11):
        print("Converting tweets of column ",i)
        tweets_to_predict = tokenizer.texts_to_matrix(final_dataset[i])
        final_dataset[i] = pd.DataFrame(clf.predict(tweets_to_predict))
    
    
      
        

