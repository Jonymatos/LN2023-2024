import pandas as pd
import numpy as np
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from custom_stop_words import chicago_hotel_names_regrex

# Download stopwords
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
stop_words = [word for word in stop_words if word not in ["not", "but", "only", "no", "against"]]
stop_words.extend(["chicago", "hotel"])

# Custom Text Preprocessor
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, hotel_names=None):
        self.stop_words = stop_words
        self.hotel_names = hotel_names
    
    def remove_stop_words(self, text):
        return " ".join([word for word in str(text).split() if word.lower() not in self.stop_words])

    def remove_hotels(self, text):
        for hotel_name in self.hotel_names:
            text = re.sub(hotel_name, "", text)
        return text
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_out = X.apply(self.remove_hotels)
        X_out = X_out.apply(self.remove_stop_words)
        return X_out

# Load data and preprocess labels
df = pd.read_csv("proj2/train.txt", sep="\t", names=["label", "review"])
mapping = {"TRUTHFULPOSITIVE": 0, "TRUTHFULNEGATIVE": 1, "DECEPTIVEPOSITIVE": 2, "DECEPTIVENEGATIVE": 3}
df.label = df.label.map(mapping)

# Construct pipeline with GridSearchCV for hyperparameter tuning
pipeline = Pipeline([
    ('preprocessor', TextPreprocessor(stop_words=stop_words, hotel_names=chicago_hotel_names_regrex)),
    ('vect', CountVectorizer(lowercase=True)),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(solver="liblinear", multi_class="auto"))
])

def crossValidation(k=10):
    param_grid = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2']
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(df.review, df.label)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}".format(grid_search.best_score_))

    # K-fold cross-validation with confusion matrix
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    aggregate_cm = np.zeros((4, 4))
    total_accuracy = 0

    for train_index, test_index in kf.split(df.review, df.label):
        X_train, X_test = df.review.iloc[train_index], df.review.iloc[test_index]
        y_train, y_test = df.label.iloc[train_index], df.label.iloc[test_index]
    
        grid_search.best_estimator_.fit(X_train, y_train)
        y_pred = grid_search.best_estimator_.predict(X_test)
    
        cm = confusion_matrix(y_test, y_pred)
        aggregate_cm += cm
    
        accuracy = accuracy_score(y_test, y_pred)
        total_accuracy += accuracy
        print(f"Fold Accuracy: {accuracy:.2f}")

    avg_cm = aggregate_cm / k
    avg_accuracy = total_accuracy / k

    print("Averaged Confusion Matrix:")
    print(avg_cm)
    print(f"\nAverage Accuracy over {k} folds: {avg_accuracy:.2f}")

def traintestsplitmethod(n_rounds=100):
    accuracy_scores = 0
    aggregate_cm = np.zeros((4, 4)) # 4x4 confusion matrix

    for i in range(n_rounds):
        X_train, X_test, y_train, y_test = train_test_split(df.review, df.label, test_size=0.1, shuffle=True, random_state=i)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        clf_accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores += clf_accuracy
        aggregate_cm += cm
        print("Round: ", i, " Accuracy: ", clf_accuracy)

    avg_cm = aggregate_cm / n_rounds
    print("Averaged Confusion Matrix:")
    print(avg_cm)
    print("Accuracy: ", accuracy_scores / n_rounds)

def evaluate_file(filename,fileout):
    # Training the classifier on entire training data
    pipeline.fit(df.review, df.label)

    # Loading and preprocessing the new reviews
    with open(filename, 'r') as file:
        new_reviews = [pipeline.named_steps['preprocessor'].remove_hotels(line.strip()) for line in file]
        new_reviews = [pipeline.named_steps['preprocessor'].remove_stop_words(line) for line in new_reviews]

    # Predicting the labels for the new reviews
    predicted_labels = pipeline.predict(pd.Series(new_reviews))
    inv_mapping = {v: k for k, v in mapping.items()}
    predicted_str_labels = [inv_mapping[label] for label in predicted_labels]

    # Writing the predicted labels to a new text file
    with open(fileout, 'w') as file:
        for label in predicted_str_labels:
            file.write(label + '\n')

if __name__ == "__main__":
    #crossValidation()
    traintestsplitmethod()
    #evaluate_file("proj2/test_just_reviews.txt","proj2/results.txt")
