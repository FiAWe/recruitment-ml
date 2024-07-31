import time
import argparse

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


from typing import Dict, Any

from model_pipeline import run_model_pipeline, fetch_data, parse_args

MODEL_NAME = 'SVM'


def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the SVM model architecture.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    # Pipeline for Naive Bayes with TF-IDF
    pipeline_svm = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC(probability=True))
    ])

    # Save variables to model_vars
    model_vars['model'] = pipeline_svm


def train_model(model_vars: Dict[str, Any]) -> None:
    """
    Train the SVM model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    model = model_vars['model']
    model.fit(model_vars['X_train'], model_vars['y_train'])


def predict(model_vars: Dict[str, Any]) -> None:
    """
    Perform predictions using the trained model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    model = model_vars['model']

    y_pred_nb = model.predict(model_vars['X_test'])
    y_pred_nb_proba = model.predict_proba(model_vars['X_test'])[:, 1]

    # Save variables to model_vars
    model_vars['y_pred_nb'] = y_pred_nb
    model_vars['y_pred_nb_proba'] = y_pred_nb_proba

if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)

