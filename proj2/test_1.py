from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "./model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("proj2/test.txt", "r") as file:
    lines = file.readlines()

# Extract labels and reviews
labels, reviews = [], []
for line in lines:
    label, review = line.split(maxsplit=1)
    labels.append(label)
    reviews.append(review)

# To store predictions and optionally compute accuracy
correct_predictions = 0

# Save the reviews in in a results.txt file
with open("results.txt", "w") as file:
    for label, review in zip(labels, reviews):
        # The model can only process 512 tokens at a time
        inputs = tokenizer(review[:512], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        
        # Optionally, compare predicted label with the true label
        if label == predicted_label:
            correct_predictions += 1

        file.write(predicted_label + "\n")

# Optionally, print the accuracy
accuracy = correct_predictions / len(labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
            