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


def predict(text: str, model_type: str = "BERT", model_dir: str = "classification_models_text_comments") -> str:
    global bert_tokenizer, bert_model, bert_classifier
    model_path = get_model_path(model_dir)
    model_file = os.path.join(model_path, "pytorch_model.bin")

    # Download model if not present
    ensure_model_downloaded(model_file, DRIVE_FILE_ID)

    # Load tokenizer and BERT base model if not already loaded
    if bert_tokenizer is None or bert_model is None:
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model.eval()

    # Load classifier model
    if bert_classifier is None:
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_classifier = BertForClassification.from_pretrained(
            pretrained_model_name_or_path=None,
            config=bert_config,
            state_dict=torch.load(model_file, map_location=torch.device('cpu'))
        )
        bert_classifier.eval()

    # Tokenize input text
    inputs = bert_tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding="max_length")

    # Predict using classifier directly on tokenized input
    with torch.no_grad():
        outputs = bert_classifier(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Rumor" if predicted_class == 1 else "Not Rumor"

