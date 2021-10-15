import sys
import pandas as pd
import re
from sqlalchemy.engine import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from custom_transformer import StartingVerbExtractor
from sklearn.svm import LinearSVC
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import nltk
nltk.download(['punkt', 'wordnet'])


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


def tokenize(text):
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

    # Stemming
    tokens = [PorterStemmer().stem(t) for t in tokens]

    # drop duplicates
    tokens = pd.Series(tokens, dtype='object').drop_duplicates().tolist()

    return tokens


def build_model():
    """
        This function will set up a pipeline which prepare for training model
    """

    pipeline = Pipeline(
        [('features', FeatureUnion([('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                                ('tfidf', TfidfTransformer())])),
                                    ('starting_verb', StartingVerbExtractor())])),
         ('clf', MultiOutputClassifier(LinearSVC()))])

    parameters = {'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
                  'features__text_pipeline__tfidf__use_idf': (True, False),
                  'features__transformer_weights': (
                        {'text_pipeline': 1, 'starting_verb': 0.5})
                  }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


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
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    import time

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        train_time = time.time() - start_time
        print("--- Traing time: %s minutes ---" % (train_time/60))

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