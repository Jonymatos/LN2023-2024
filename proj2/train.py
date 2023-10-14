import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

data = pd.read_csv("proj2/train.txt", sep="\t", names=["label", "review"])

mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}

data["label"] = data["label"].map(mapping)

data = data.to_dict(orient="records")

model_name = "distilbert-base-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=4)

def tokenize(batch):
    return tokenizer(batch["review"], padding=True, truncation=True)

train_dataset = tokenize(data)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
)

def format_labels(batch):
    return {"labels": batch["label"]}

train_dataset = train_dataset.map(format_labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

trainer.save_model("./model")