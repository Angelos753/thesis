import os
os.environ["USE_TF"] = "0"

import torch
import gdown
from transformers import BertTokenizer, BertModel, BertConfig
from model_def import BertForClassification
import numpy as np

bert_tokenizer = None
bert_model = None
bert_classifier = None

# Google Drive ID for the shared model file
DRIVE_FILE_ID = "1AxPXc2rq-mDm6OCY-DLiVdIQpCFmSxB4"


def ensure_model_downloaded(local_path: str, drive_id: str):
    if not os.path.exists(local_path):
        print(f"\n🧰 Downloading model to {local_path} from Google Drive...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, local_path, quiet=False)


def get_model_path(model_dir):
    return os.path.join("./trained_models", model_dir)


def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
    return embeddings.squeeze(0).numpy()


def predict(text: str, model_type: str = "BERT", model_dir: str = "classification_models_text_comments") -> str:
    global bert_tokenizer, bert_model, bert_classifier
    model_path = get_model_path(model_dir)
    model_file = os.path.join(model_path, "pytorch_model.bin")

    ensure_model_downloaded(model_file, DRIVE_FILE_ID)

    if bert_tokenizer is None:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model.eval()

    if bert_classifier is None:
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
            logits = outputs[0]
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

    with torch.no_grad():
        outputs = bert_classifier(torch.tensor(embeddings))
        logits = outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Rumor" if predicted_class == 1 else "Not Rumor"
