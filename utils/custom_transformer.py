import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Tokenizer(BaseEstimator, TransformerMixin):
    def tokenize(self, text):
        """
            This function will normalize, removed stop words, stemmed and lemmatized.
            Returns tokenized text
        """

        # Normalize text
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

        # Tokenize text
        tokens = word_tokenize(text)

        # Remove stop words
        tokens = [t for t in tokens if t not in stopwords.words("english")]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        # reduce words to their root form using default pos
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        # lemmatize verbs by specifying pos
        tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
        # lemmatize verbs by specifying pos
        tokens = [lemmatizer.lemmatize(t, pos='a') for t in tokens]

        # drop duplicates
        tokens = list(pd.Series(tokens, dtype='object').drop_duplicates().values)

        return ' '.join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(self.tokenize).values


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def tokenize(self, text):
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(self.tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP', 'VBG'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


