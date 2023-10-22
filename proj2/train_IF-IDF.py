import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk, re
from sklearn.model_selection import train_test_split
from custom_stop_words import chicago_hotel_names_regrex
from sklearn.svm import SVC

ROUNDS = 100

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
stop_words.append("chicago")
stop_words.append("hotel")


def remove_stop_words(text):
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])

def remove_hotels(text):
    for hotel_name in chicago_hotel_names_regrex:
        text = re.sub(hotel_name, "", text)
    return text

df.review = df.review.apply(remove_hotels)
df.review = df.review.apply(remove_stop_words)

LRC = ("clf", LogisticRegression(solver="liblinear", multi_class="auto"))
SVC = ("clf", SVC(kernel="linear", gamma="auto"))

clf = Pipeline([
    ("vect", CountVectorizer(lowercase=True)),
    ("tfidf", TfidfTransformer()),
    LRC
])

accuracy_scores = []

for i in range(ROUNDS):
    X_train, X_test, y_train, y_test = train_test_split(df.review, df.label, test_size=0.1, shuffle=True, random_state=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf_accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(clf_accuracy)
    #print("Round: ", i, " Accuracy: ", clf_accuracy)

print("Accuracy: ", sum(accuracy_scores) / len(accuracy_scores))


