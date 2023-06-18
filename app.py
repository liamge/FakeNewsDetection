import transformers

import numpy as np

from flask import Flask, request, jsonify, render_template
from tokenizers import BertWordPieceTokenizer
from model import model_predict, load_model, StringCleaner, Stemmer

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
    app.run(host='0.0.0.0', debug=True)