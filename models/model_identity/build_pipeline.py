from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from models.model_identity.custom_transformer import Tokenizer, StartingVerbExtractor


def build_pipeline():
    """
        This function will set up a pipeline which prepare for training model
    """

    pipeline = Pipeline([('features', FeatureUnion(
        [('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=Tokenizer().transform)),
                                     ('tfidf', TfidfTransformer())])),
         ('starting_verb', StartingVerbExtractor())])),
                         ('clf', MultiOutputClassifier(LinearSVC()))])

    return pipeline