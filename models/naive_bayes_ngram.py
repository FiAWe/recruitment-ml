from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import Dict, Any

from model_pipeline import run_model_pipeline, fetch_data, parse_args

# Training and prediction functions are same as base Naive Bayes model
from naive_bayes import train_model, predict

MODEL_NAME = 'naive_bayes_ngram'

def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the model pipeline.
    We will use the Multinomial Naive Bayes model with TF-IDF vectorizer.
    This version of the model uses n-grams (1, 3) for feature extraction.
    n-grams are contiguous sequences of n items from a given sample of text or speech.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
        ('nb', MultinomialNB())
    ])
    model_vars['pipeline'] = pipeline_nb

if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)

