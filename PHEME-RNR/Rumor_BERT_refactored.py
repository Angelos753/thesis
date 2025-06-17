
"""
Refactored Rumor Detection BERT Script
- Simplified for clarity and maintainability
- Uses HuggingFace Transformers and PyTorch
- Prepares and trains a BERT model for text classification
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load and preprocess data
raw_data = pd.read_csv('./data/raw_data.csv')
raw_data = raw_data[['text_comments', 'label']].dropna()
raw_data = raw_data.rename(columns={'text_comments': 'text'})

# Encode labels
raw_data['label'] = LabelEncoder().fit_transform(raw_data['label'])

# Train-test split
train_df, val_df = train_test_split(raw_data, test_size=0.2, random_state=42)

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and tokenize dataset
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
)

# Trainer setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save model
model.save_pretrained("./trained_models/classification_models_text_comments")
tokenizer.save_pretrained("./trained_models/classification_models_text_comments")
