import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec


from typing import Dict, Any

from model_pipeline import run_model_pipeline, fetch_data, parse_args

# Training and prediction functions are same as base Naive Bayes model
from naive_bayes import train_model, predict

MODEL_NAME = 'naive_bayes_Word2Vec'

# Create a custom transformer to convert text to Word2Vec vectors
# Worked from: https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673owardsdatascience.com/multi-class-text-classification-with-word2vec-and-deep-learning-e1d19029df5f
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, X: np.ndarray, y=None):
        sentences = [text.split() for text in X]
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X: np.ndarray):
        sentences = [text.split() for text in X]
        return np.array([np.mean([self.model.wv[word] for word in sentence if word in self.model.wv] or [np.zeros(self.vector_size)], axis=0) for sentence in sentences])

def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the model pipeline.
    We will use the Multinomial Naive Bayes model with Word2Vec vectorizer.
    Word2Vec is a word embedding technique that converts words to vectors by
    capturing the context in which the word appears in the text.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    pipeline_nb = Pipeline([
        ('word2vec', Word2VecTransformer(vector_size=100, window=50, min_count=1, workers=4)),
        ('nb', GaussianNB())
    ])

    model_vars['pipeline'] = pipeline_nb

if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)

