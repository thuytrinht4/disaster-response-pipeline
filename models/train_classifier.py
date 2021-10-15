import sys
import pandas as pd
from sqlalchemy.engine import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import joblib

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils.custom_transformer import Tokenizer, StartingVerbExtractor



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


def build_pipeline():
    """
        This function will set up a pipeline which prepare for training model
    """

    pipeline = Pipeline([('features', FeatureUnion(
        [('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=Tokenizer().transform)),
                                     ('tfidf', TfidfTransformer())])),
         ('starting_verb', StartingVerbExtractor())])),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {  'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
                    'clf__estimator__n_estimators': [10, 20, 50]
              }
    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=3)

    return cv


def load_data(database_filepath):
    """
        This function will load the dataset from a database_path in the processing stage,
        Return
            X: dataframe of messages
            y : dataframe of 36 categories output
            category_names : name of 36 categories
    """

    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    # engine = create_engine('sqlite:///InsertDatabaseName.db')
    # df = pd.read_sql_table(database_filepath, engine)
    X = df['message'].values
    y = df.iloc[:, 4:].values
    category_names = list(df.iloc[:, 4:].columns)

    return X, y, category_names

def evaluate_model(model, X_test, Y_test, category_names):
    """
        This function will print out the information of accuracy, precision, recall scores of 36 categories
    """
    y_pred = model.predict(X_test)

    for k in range(len(category_names)):
        print('Number of category ', k, '\t name: ', category_names[k], '.\n')
        print('\t Accuracy = ', (y_pred[:, k] == Y_test[:, k]).mean(),
              '\t % Precision: \t', precision_score(Y_test[:, k], y_pred[:, k]),
              '\t % Recall : \t', recall_score(Y_test[:, k], y_pred[:, k]),
              '\t % F1_score : \t', f1_score(Y_test[:, k], y_pred[:, k])
              )

def save_model(model, model_filepath):
    """ Save model's best_estimator_ using pickle"""
    # pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    joblib.dump(model, open(model_filepath, 'wb'))

def main():
    import time

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

        print('Building model...')
        model = build_pipeline()

        print('Training model...')
        start_time = time.time()
        print(f"--- Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))} ---")
        model.fit(X_train, Y_train)
        train_time = time.time() - start_time
        print("--- Training time: %s minutes ---" % (train_time/60))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()