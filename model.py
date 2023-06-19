import pickle
import re
import string

import transformers

import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

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

def load_model(model_path, model_type):
    if model_type == "bert":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, 
                                    custom_objects={"TFDistilBertModel": transformers.TFDistilBertModel})
    elif model_type == "tinybert":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, 
                                    custom_objects={"TFBertModel": transformers.TFBertModel})
    elif model_type == 'tflite':
        import tensorflow as tf
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
    elif model_type == 'sklearn':
        model = pickle.load(open(model_path, 'rb'))
    
    return model

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    all_ids = []
    
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def model_predict(input, model, model_type, tokenizer=None):
    if model_type == "tensorflow":
        if tokenizer != None:
            encoded = fast_encode(np.array(input), tokenizer, maxlen=192)
            pred = model.predict(encoded)
            return np.round(pred), np.round(pred, decimals=4)
        else:
            print("Error: No tokenizer passed but BERT model specified")
    elif model_type == "tflite":
        if tokenizer != None:
            encoded = fast_encode(np.array(input), tokenizer, maxlen=192)
            input_data = encoded.astype(np.int32)

            input_details = model.get_input_details()
            output_details = model.get_output_details()

            # Test model on random input data.
            model.set_tensor(input_details[0]['index'], input_data)

            model.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = model.get_tensor(output_details[0]['index'])
            return np.round(output_data), np.round(output_data, decimals=4)
        else:
            print("Error: No tokenizer passed but BERT model specified")
    elif model_type == "sklearn":
        pred = model.predict(input)
        proba = model.predict_proba(input)
        return np.round(pred), np.round(proba, decimals=4)
