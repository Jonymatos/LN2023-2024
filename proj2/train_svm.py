import pandas as pd
import numpy as np
import nltk, re
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from custom_stop_words import chicago_hotel_names_regrex

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
np.random.seed(500)

Corpus = pd.read_csv("proj2/train.txt", sep="\t", names=["label", "review"],encoding='latin-1')

mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}
inverse_mapping = {v: k for k, v in mapping.items()}

Corpus["label"] = Corpus["label"].map(mapping)

stop_words = stopwords.words('english')
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

Corpus.review = Corpus.review.apply(remove_hotels)
#Corpus.review = Corpus.review.apply(remove_stop_words)

# Step - a : Remove blank rows if any.
Corpus['review'].dropna(inplace=True)
# Step - b : Change all the reviews to lower case. This is required as python interprets 'dog' and 'DOG' differently
#Corpus['review'] = [entry.lower() for entry in Corpus['review']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
Corpus['review']= [word_tokenize(entry) for entry in Corpus['review']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(Corpus['review']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'review_pre'
    Corpus.loc[index,'review_pre'] = str(Final_words)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['review_pre'],Corpus['label'],test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['review_pre'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

