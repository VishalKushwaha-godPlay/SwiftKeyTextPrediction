# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['experience'].fillna(0, inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(), inplace=True)

X = dataset.iloc[:, :3]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
##model = pickle.load(open('model.pkl','rb'))
##print(model.predict([[2, 9, 6]]))

import re 
from nltk.tokenize import word_tokenize
def extra_space(text):
    new_text= re.sub("\s+"," ",text)
    return new_text
def sp_charac(text):
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text
def tokenize_text(text):
    new_text=word_tokenize(text)
    return new_text

"""### Predicting Next Word"""

def predict_next():
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    import pickle
    import time
  
    with open('tokenizer1_len7.pickle', 'rb') as handle:
        tokenizer_len7 = pickle.load(handle)

    with open('tokenizer1_len4.pickle', 'rb') as handle:
        tokenizer_len4 = pickle.load(handle)
        
    with open('tokenizer1_len2.pickle', 'rb') as handle:
        tokenizer_len2 = pickle.load(handle)
    
    file="lstm_len7.hdf5"
    model_len7 = load_model(file)
    model_len7.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    
    file="lstm_len4.hdf5"
    model_len4 = load_model(file)
    model_len4.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    file="lstm_len2.hdf5"
    model_len2 = load_model(file)
    model_len2.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    while(True):
        text=input()
        start= time.time()
        cleaned_text=extra_space(text)
        cleaned_text=sp_charac(cleaned_text)
        tokenized=tokenize_text(cleaned_text)
    
        line = ' '.join(tokenized)
        
    
        if len(tokenized)==1:
        
            encoded_text = tokenizer_len2.texts_to_sequences([line])
            pad_encoded = pad_sequences(encoded_text, maxlen=1, truncating='pre')
    
            for i in (model_len2.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    
    
                pred_word = tokenizer_len2.index_word[i]
                print("Next word suggestion:",pred_word)
        elif len(tokenized)<4:
            encoded_text = tokenizer_len4.texts_to_sequences([line])
            pad_encoded = pad_sequences(encoded_text, maxlen=3, truncating='pre')
    
            for i in (model_len4.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    
    
                pred_word = tokenizer_len4.index_word[i]
                print("Next word suggestion:",pred_word)
        else:
            encoded_text = tokenizer_len7.texts_to_sequences([line])
            pad_encoded = pad_sequences(encoded_text, maxlen=6, truncating='pre')
    
            for i in (model_len7.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    
    
                pred_word = tokenizer_len7.index_word[i]
                print("Next word suggestion:",pred_word)
        print('Time taken: ',time.time()-start)

    model=predict_next()
    return model