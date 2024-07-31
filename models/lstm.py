import time
import argparse
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D

from typing import Dict, Any

from model_pipeline import run_model_pipeline, fetch_data, parse_args

MODEL_NAME = 'LSTM'


def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the LSTM model architecture.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    # Tokenisation
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(model_vars['X_train'])
    X_train_seq = tokenizer.texts_to_sequences(model_vars['X_train'])
    X_test_seq = tokenizer.texts_to_sequences(model_vars['X_test'])

    # Padding
    max_len = 300
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    # LSTM Model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    # Save variables to model_vars
    model_vars['model'] = model
    model_vars['X_train_pad'] = X_train_pad
    model_vars['X_test_pad'] = X_test_pad

def train_model(model_vars: Dict[str, Any]) -> None:
    """
    Train the LSTM model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    model = model_vars['model']
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(model_vars['X_train_pad'], model_vars['y_train'], epochs=10, batch_size=64)


def predict(model_vars: Dict[str, Any]) -> None:
    """
    Perform predictions using the trained LSTM model.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """
    model = model_vars['model']
    y_pred_nb_proba = model.predict(model_vars['X_test_pad'])
    y_pred_nb_proba = y_pred_nb_proba.flatten()
    y_pred_nb = np.where(y_pred_nb_proba > 0.5, 1, 0)

    # Save variables to model_vars
    model_vars['y_pred_nb_proba'] = y_pred_nb_proba
    model_vars['y_pred_nb'] = y_pred_nb

if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)

