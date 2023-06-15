import numpy as np
from tqdm import tqdm

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

def model_predict(input, model, tokenizer):
    encoded = fast_encode(np.array(input), tokenizer, maxlen=192)
    pred = model.predict(encoded)
    return np.round(pred), np.round(pred, decimals=2)