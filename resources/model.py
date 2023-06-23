# # Libraries for data loading, data manipulation and data visualisation
import numpy as np
import pandas as pd
import re

import nltk

from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle


df1 = pd.read_csv("resources/train copy.csv")
df2 = pd.read_csv("resources/train.csv") # Read in csv file as pandas dataframe

df_train = pd.concat([df1, df2], ignore_index=True) # Concatenate the two dataframes
df_train.head() # Check the first few rows of the dataframe

pattern_url = [r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', r'@\w+\b', r'RT', r'\#\w+\b', r"\b(?:climate change|global warming)\b"]

df_train['message'] = df_train['message'].replace(to_replace = pattern_url, value = '', regex = True)


def remove_non_english_words(text):
    """
    
    """
    # Regular expression pattern to match English words and apostrophes
    pattern = re.compile(r"\b[a-zA-Z']+")

    # Find all English words and apostrophes in the text
    english_words = re.findall(pattern, text)

    # Join the English words into a single string
    result = ' '.join(english_words)

    return result

df_train['message'] = df_train['message'].apply(remove_non_english_words)


df_train['message'] = df_train['message'].str.lower()


def remove_https(text):
    return re.sub(r'https', ' ', text)

df_train['message'] = df_train['message'].apply(remove_https)


df_train = df_train.drop_duplicates(keep='first')

from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()

def text_processing(text):
    text = nltk.word_tokenize(text.lower())
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            lemming = lem.lemmatize(i)
            y.append(lemming)
            
    return " ".join(y)

df_train['message_transformed'] = df_train['message'].apply(text_processing)

X = df_train['message_transformed']
y = df_train['sentiment']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm = SVC()
svm.fit(X_train_vec, y_train)

svm_y_pred = svm.predict(X_test_vec)
print(classification_report(y_test, svm_y_pred))

with open("resources/svm.pkl", 'wb') as file:
    pickle.dump(svm, file)

with open("resources/vectorizer.pkl", 'wb') as file:
    pickle.dump(vectorizer, file)












