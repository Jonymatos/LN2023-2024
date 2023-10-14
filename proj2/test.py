from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd

model_name = "./model" 
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

with open("proj2/test_just_reviews.txt", "r") as f:
    data = f.readlines()

test_data = tokenizer(data, padding=True, truncation=True)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_data["input_ids"]),
    torch.tensor(test_data["attention_mask"])
)

predictions = model(test_dataset.tensors[0], attention_mask=test_dataset.tensors[1])
predicted_labels = torch.argmax(predictions.logits, axis=1).tolist()

mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}

predicted_labels = [mapping[prediction] for prediction in predicted_labels]

# Save the predicted labels to a file
with open("predicted_labels.txt", "w") as file:
    for label in predicted_labels:
        file.write(label + "\n")