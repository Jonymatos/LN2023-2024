import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

data = pd.read_csv("proj2/train_split.txt", sep="\t", names=["label", "review"])

mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}
inverse_mapping = {v: k for k, v in mapping.items()}

data["label"] = data["label"].map(mapping)

#data = data.to_dict(orient="records")
hg_dataset = Dataset(pa.Table.from_pandas(data))


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model_name = "distilbert-base-cased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4, id2label=inverse_mapping, label2id=mapping)

def tokenize(batch):
    return tokenizer(batch["review"], truncation=True, padding=True)

train_dataset = hg_dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("./model")


