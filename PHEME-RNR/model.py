import os
os.environ["USE_TF"] = "0"

import torch
import gdown
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, BertModel, BertConfig
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, LSTM, Dense
from model_def import BertForClassification
from transformer import TransformerLayer

bert_tokenizer = None
bert_model = None
bert_classifier = None

# Google Drive ID for the shared model file
DRIVE_FILE_ID = "1AxPXc2rq-mDm6OCY-DLiVdIQpCFmSxB4"

def ensure_model_downloaded(local_path: str, drive_id: str):
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
        print(f"\n🧰 Downloading model to {local_path} from Google Drive...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}&confirm=t"
        print(f"\n URL: {url}")
        gdown.download(url, local_path, quiet=False, fuzzy=True)

def get_model_path(model_dir):
    return os.path.join("./trained_models", model_dir)

def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings.squeeze(0).numpy()

def build_lstm_model_from_dir(model_dir):
    model_path = os.path.join(get_model_path(model_dir), "LSTM_model/model.h5")
    text_input = Input(shape=(None, 768), dtype='float32', name='text')
    l_mask = Masking(mask_value=-99.)(text_input)
    encoded = LSTM(100)(l_mask)
    dense = Dense(30, activation='relu')(encoded)
    output = Dense(2, activation='softmax')(dense)
    model = Model(text_input, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.load_weights(model_path)
    return model

def build_transformer_model_from_dir(model_dir):
    model_path = os.path.join(get_model_path(model_dir), "Transformer_model/model.h5")
    embed_dim = 768
    ff_dim = 32
    num_heads = 1
    text_input = Input(shape=(None, 768), dtype='float32', name='text')
    l_mask = Masking(mask_value=-99.)(text_input)
    encoded = TransformerLayer(embed_dim, num_heads, ff_dim)(l_mask)
    lstm_out = LSTM(100)(encoded)
    dense = Dense(30, activation='relu')(lstm_out)
    output = Dense(2, activation='softmax')(dense)
    model = Model(text_input, output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.load_weights(model_path)
    return model

def predict(text: str, model_type: str = "BERT", model_dir: str = "classification_models_text_comments") -> str:
    global bert_tokenizer, bert_model, bert_classifier
    model_path = get_model_path(model_dir)
    model_file = os.path.join(model_path, "pytorch_model.bin")

    if bert_tokenizer is None:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model.eval()

    if model_type == "BERT" and bert_classifier is None:
        ensure_model_downloaded(model_file, DRIVE_FILE_ID)
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_classifier = BertForClassification.from_pretrained(
            pretrained_model_name_or_path=None,
            config=bert_config,
            state_dict=torch.load(model_file, map_location=torch.device('cpu'))
        )
        bert_classifier.eval()

    if model_type == "BERT":
        inputs = bert_tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = bert_classifier(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            predicted_class = torch.argmax(logits, dim=1).item()
        return "Rumor" if predicted_class == 1 else "Not Rumor"

    embeddings = get_bert_embeddings(text)
    max_len = 200
    if embeddings.shape[0] > max_len:
        embeddings = embeddings[:max_len]
    else:
        pad_len = max_len - embeddings.shape[0]
        padding = np.full((pad_len, 768), -99., dtype=np.float32)
        embeddings = np.vstack((embeddings, padding))
    embeddings = np.expand_dims(embeddings, axis=0)

    if model_type == "LSTM":
        prediction = build_lstm_model_from_dir(model_dir).predict(embeddings)[0]
    elif model_type == "Transformer":
        prediction = build_transformer_model_from_dir(model_dir).predict(embeddings)[0]
    else:
        raise ValueError("Unknown model type")

    predicted_class = np.argmax(prediction)
    return "Rumor" if predicted_class == 1 else "Not Rumor"
