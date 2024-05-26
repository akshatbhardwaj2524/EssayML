from flask import Flask, request, jsonify
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from gensim.models import KeyedVectors
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)

def sent2word(x):
    stop_words = set(stopwords.words('english')) 
    x = re.sub("[^A-Za-z]", " ", x)
    x = x.lower()
    filtered_sentence = [] 
    words = x.split()
    for w in words:
        if w not in stop_words: 
            filtered_sentence.append(w)
    return filtered_sentence

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words = []
    for i in raw:
        if len(i) > 0:
            final_words.append(sent2word(i))
    return final_words

def makeVec(words, model, num_features):
    vec = np.zeros((num_features,), dtype="float32")
    noOfWords = 0.
    index2word_set = set(model.wv.index_to_key)
    for i in words:
        if i in index2word_set:
            noOfWords += 1
            vec = np.add(vec, model.wv[i])        
    vec = np.divide(vec, noOfWords)
    return vec

def getVecs(essays, model, num_features):
    essay_vecs = []
    for essay in essays:
        essay_vecs.append(makeVec(essay, model, num_features))
    return np.array(essay_vecs)

def get_model():
    model = load_model('final_lstm.h5')
    return model

def convertToVec(text):
    model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)
    num_features = 300
    clean_test_essays = [sent2word(text)]
    testDataVecs = getVecs(clean_test_essays, model, num_features)
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
    lstm_model = get_model()
    preds = lstm_model.predict(testDataVecs)
    return str(round(preds[0][0]))

@app.route('/', methods=['GET', 'POST'])  # Allow both GET and POST requests
def create_task():
    if request.method == 'POST':
        final_text = request.json["text"]
        score = convertToVec(final_text)
        return jsonify({'score': score}), 201
    else:
        return "Welcome to Essay Scoring API!"

if __name__ == '__main__':
    app.run(debug=True)
