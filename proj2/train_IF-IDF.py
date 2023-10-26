import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk, re
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from custom_stop_words import chicago_hotel_names_regrex
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import numpy as np

ROUNDS = 100

df = pd.read_csv("proj2/train.txt", sep="\t", names=["label", "review"])

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
stop_words.remove("didn't")
stop_words.remove("wasn't")
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

accuracy_scores = 0
aggregate_cm = np.zeros((4, 4)) # 4x4 confusion matrix


for i in range(ROUNDS):
    X_train, X_test, y_train, y_test = train_test_split(df.review, df.label, test_size=0.1, shuffle=True, random_state=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    clf_accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores += clf_accuracy
    aggregate_cm += cm
    print("Round: ", i, " Accuracy: ", clf_accuracy)

# Average the confusion matrix over k-folds
avg_cm = aggregate_cm / ROUNDS

print("Averaged Confusion Matrix:")
print(avg_cm)
print("Accuracy: ", accuracy_scores / ROUNDS)


