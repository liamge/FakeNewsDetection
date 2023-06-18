import transformers

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask, request, jsonify, render_template
from tokenizers import BertWordPieceTokenizer
from model import model_predict, load_model

class StringCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        X = [self.clean_string(x) for x in X]
        return X
    
    def replace_url(self, s):
        return re.sub(r'http\S+', ' URL ', s)

    def replace_mentions(self, s):
        return re.sub(r'@([A-Za-z0-9_]+)', ' MENTION ')

    def replace_nums(self, s):
        return re.sub(r'\d+', ' NUM ', s)

    def remove_punct(self, s):
        return ''.join(x for x in s if x not in string.punctuation)

    def whitespace_regularization(self, s):
        # If there is more than one space in a row in our string, 
        # we normalize that to one space
        return re.sub(r'\s+', ' ', s)

    def clean_string(self, s):
        temp = self.replace_url(s.lower())
        temp = self.replace_nums(temp)
        temp = self.remove_punct(temp)
        temp = self.whitespace_regularization(temp)

        return temp
    
class Stemmer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        stemmer = PorterStemmer()
        X = [' '.join([stemmer.stem(w) for w in word_tokenize(x)]) for x in X]
        return X

app = Flask(__name__)
model = load_model('models/pipe.pkl', 'sklearn')

#tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
#tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
#fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    text_input = list(request.form.values())
    pred_class, probability = model_predict(text_input, model, 
                                            model_type='sklearn', tokenizer=fast_tokenizer)

    if pred_class.shape == (1,1):
        if pred_class[0][0] == 0:
            prediction = "Fake News"
            probability = 1 - probability
        elif pred_class[0][0] == 1:
            prediction = "True News"
    elif pred_class.shape == (1,):
        if pred_class[0] == 0:
            prediction = "Fake News"
            probability = 1 - probability
        elif pred_class[0] == 1:
            prediction = "True News"

    return render_template('index.html', prediction_text='Predicted {} with probability {}'.format(prediction, probability[0][0]))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    text_input = list(data.values())
    pred_class, _ = model_predict(text_input, model, 
                                  model_type='sklearn', tokenizer=fast_tokenizer)

    if pred_class.shape == (1,1):
        if pred_class[0][0] == 0:
            prediction = "Fake News"
            probability = 1 - probability
        elif pred_class[0][0] == 1:
            prediction = "True News"
    elif pred_class.shape == (1,):
        if pred_class[0] == 0:
            prediction = "Fake News"
            probability = 1 - probability
        elif pred_class[0] == 1:
            prediction = "True News"

    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run()