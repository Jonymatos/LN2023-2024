from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "./model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


with open("test_just_reviews.txt", "r") as file:
    reviews = file.readlines()

#Save the reviews in in a results.txt file
with open("results.txt", "w") as file:
        for review in reviews:
            #Unfortunatly, the model can only process 512 tokens at a time
            inputs = tokenizer(review[:512], return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            file.write(model.config.id2label[predicted_class_id] + "\n")
            