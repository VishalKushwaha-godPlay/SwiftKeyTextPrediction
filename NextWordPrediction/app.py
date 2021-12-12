import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.form.get("data")
    prediction = predict_next(data)
    output = prediction

    return render_template('index.html', prediction_text='Next word suggestions are: {}'.format(output))

@app.route('/predict_next',methods=['POST'])
def predict_next(data):
    
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
   
    text=data
    start= time.time()
    cleaned_text=extra_space(text)
    cleaned_text=sp_charac(cleaned_text)
    tokenized=tokenize_text(cleaned_text)
    
    line = ' '.join(tokenized)
    
    if len(tokenized)==1:
    
        encoded_text = tokenizer_len2.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=1, truncating='pre')
        final1=''
        for i in (model_len2.predict(pad_encoded)[0]).argsort()[-3:][::-1]:

            pred_word1 = tokenizer_len2.index_word[i]
            print("Next word suggestion:",pred_word1)
            final1=pred_word1+" ,"+final1
            
        return  final1 
    
    elif len(tokenized)<4:
        encoded_text = tokenizer_len4.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=3, truncating='pre')
        final2=''
        for i in (model_len4.predict(pad_encoded)[0]).argsort()[-4:][::-1]:


            pred_word2 = tokenizer_len4.index_word[i] 
            print("Next word suggestion:",pred_word2)
            final2=pred_word2+" ,"+final2
        
        return final2
        
    else:
        encoded_text = tokenizer_len7.texts_to_sequences([line])
        pad_encoded = pad_sequences(encoded_text, maxlen=6, truncating='pre')
        final3=''
        for i in (model_len7.predict(pad_encoded)[0]).argsort()[-5:][::-1]:

            pred_word3 = tokenizer_len7.index_word[i]
            print("Next word suggestion:",pred_word3)
            final3=pred_word3+" ,"+final3
            
        return final3
            
    print('Time taken: ',time.time()-start)
        
    

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = predict_next(data)

    output = prediction
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)