from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from typing import Dict, Any
from pandas import Series

from model_pipeline import run_model_pipeline, fetch_data, parse_args

MODEL_NAME = 'naive_bayes'

def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the model pipeline.
    We will use the Multinomial Naive Bayes model with TF-IDF vectorizer.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    pipeline_nb = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])
    model_vars['pipeline'] = pipeline_nb


def train_model(model_vars: Dict[str, Any]) -> None:
    """
    Train the model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    pipeline_nb: Pipeline = model_vars['pipeline']
    X_train: Series = model_vars['X_train']
    y_train: Series = model_vars['y_train']

    pipeline_nb.fit(X_train, y_train)


def predict(model_vars: Dict[str, Any]) -> None:
    """
    Perform predictions using the trained model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    pipeline_nb: Pipeline = model_vars['pipeline']
    X_test: Series = model_vars['X_test']
    
    y_pred_nb = pipeline_nb.predict(X_test)
    y_pred_nb_proba = pipeline_nb.predict_proba(X_test)[:, 1]

    # Save variables to model_vars
    model_vars['y_pred_nb'] = y_pred_nb
    model_vars['y_pred_nb_proba'] = y_pred_nb_proba


if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)

