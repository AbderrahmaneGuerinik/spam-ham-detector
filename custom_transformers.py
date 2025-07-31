import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import spacy


tokenizer = joblib.load('./models/tokenizer.joblib')
model_embeeding = joblib.load('./models/word2vec.joblib')

nlp = spacy.load("en_core_web_sm")


# utilities functions
def num_digits(text):
    matches = re.findall('\d', text)
    return len(matches)


def has_link(text):
    match = re.search(
        r'[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&//=]*)', text)
    if bool(match):
        return 1
    else:
        return 0


def has_email(text):
    match = re.search(
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text)
    if bool(match):
        return 1
    else:
        return 0


def is_question(text):
    match = re.search(r'\?$', text.strip())
    if bool(match):
        return 1
    else:
        return 0


spam_keywords = [
    "free", "win", "winner", "cash", "prize", "credit", "urgent",
    "claim", "offer", "now", "limited", "click", "buy", "cheap",
    "guaranteed", "access", "trial", "money", "investment", "deal",
    "loan", "congratulations", "bonus", "discount", "promo",
    "act now", "don’t delete", "this won’t last", "exclusive"
]


def contain_spam_keywords(text):
    i = 0
    while (i < len(spam_keywords)):
        if spam_keywords[i] in text.lower():
            break
        i += 1
    if i < len(spam_keywords):
        return 1
    else:
        return 0


def spam_words_count(text):
    i = 0
    for word in spam_keywords:
        if word in text.lower():
            i += 1
    return i


greetings = [
    "hi", "hello", "dear", "greetings", "hey", "good morning", "good afternoon", "good evening",
    "what's up", "yo", "howdy", "hiya", "sup", "to whom it may concern", "dearest",
    "dear friend", "dear customer", "dear user", "attention", "dear valued customer"
]


def stats_with_greeting(text):
    text = text.lower().strip()
    return int(any(text.startswith(greet) for greet in greetings))


signature_closings = [
    "regards",
    "best regards",
    "kind regards",
    "warm regards",
    "with regards",
    "sincerely",
    "yours sincerely",
    "yours truly",
    "truly yours",
    "faithfully yours",
    "with gratitude",
    "thank you",
    "thanks",
    "many thanks",
    "thanks in advance",
    "cheers",
    "respectfully",
    "warm wishes",
    "best wishes",
    "all the best",
    "have a great day",
    "take care",
    "talk soon",
    "stay safe"
]


def has_closing(text):
    text = text[-100:]
    return int(any(closing in text.lower().strip() for closing in signature_closings))


def num_special_chars(text):
    matches = re.findall(r'[^a-zA-Z0-9\s\.,]', text.lower().strip())
    return len(matches)


def num_exclamation_marks(text):
    return text.count('!')


def number_upper_words(text):
    words = text.split()
    return sum(1 for word in words if word.isupper())


def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


tokenizer = tokenizer.from_pretrained('gpt2')


def tokenize(text):
    return tokenizer.tokenize(text)


def vectorize_sentence(sentence):
    vectors = [model_embeeding.wv[token]
               for token in sentence if token in model_embeeding.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model_embeeding.vector_size)


# Create the estimators
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['message_length'] = df['message'].apply(len)
        df['num_digits'] = df['message'].apply(num_digits)
        df['has_link'] = df['message'].apply(has_link)
        df['has_email'] = df['message'].apply(has_email)
        df['is_question'] = df['message'].apply(is_question)
        df['contain_spam_keywords'] = df['message'].apply(
            contain_spam_keywords)
        df['spam_keywords_count'] = df['message'].apply(spam_words_count)
        df['starts_with_greeting'] = df['message'].apply(stats_with_greeting)
        df['has_closing'] = df['message'].apply(has_closing)
        df['num_sepcail_chars'] = df['message'].apply(num_special_chars)
        df['num_excalamation_marks'] = df['message'].apply(
            num_exclamation_marks)
        df['number_upper_words'] = df['message'].apply(number_upper_words)
        return df


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = X.apply(self.clean)
        return np.vstack(vectors.values)

    def clean(self, text):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = lemmatize_text(text)
        tokens = tokenize(text)
        vector = vectorize_sentence(tokens)
        return vector

    def get_feature_names_out(self, input_features=None):
        return [f'vector_{i}' for i in range(100)]


class Renamer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3',
                              'Unnamed: 4'], axis=1)
        df = df.rename(columns={'v1': 'label', 'v2': 'message'})
        return df
