import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import sys
from custom_stop_words import chicago_hotel_names

df = pd.read_csv("train.txt", sep="\t", names=["label", "review"])

mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}
inv_mapping = {v: k for k, v in mapping.items()}
df.label = df.label.map(mapping)

#Stop words
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
#Remove but and not from stop words
stop_words.remove("not")
stop_words.remove("but")
stop_words.remove("only")
stop_words.remove("no")
stop_words.remove("against")


def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words and word.lower() not in chicago_hotel_names])

def remove_hotels(text):
    #See words two by two
    words = text.split()
    new_words = []
    for i in range(len(words)-1):
        if words[i] + " " + words[i+1] not in chicago_hotel_names:
            new_words.append(words[i])
    return " ".join(new_words)

df.review = df.review.apply(remove_stop_words)
inv_mapping[df["label"][6]], df["review"][6]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.review, df.label, test_size=0.1, shuffle=True)

clf = Pipeline([
    ("vect", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("clf", LogisticRegression())
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#Get review that were wrongly classified
wrongly_classified = X_test[y_pred != y_test]

classifier_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of classifier: {:.4f}".format(classifier_accuracy))

#Test the following review
review = "The staff was very rude and unhelpful. The room was dirty and the bathroom was not clean. I will not stay here again."

#Predict the sentiment
sentiment = clf.predict([review])
print("Sentiment test review: {}".format(inv_mapping[sentiment[0]]))
print()

for wrongly_classified_review in wrongly_classified:
    sentiment = clf.predict([wrongly_classified_review])
    print("Sentiment: {}".format(inv_mapping[sentiment[0]]))
    print()
    print("Review: {}".format(wrongly_classified_review))
    print()


