import tensorflow as tf
import numpy as np

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight

from typing import Dict, Any

from model_pipeline import run_model_pipeline, fetch_data, parse_args


MODEL_NAME = 'BERT'

def declare_model(model_vars: Dict[str, Any]) -> None:
    """
    Declare and configure the model architecture.
    We will use the DistilBERT model for sequence classification.
    We are using Adam optimizer with an initial learning rate of 2e-5.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.

    Returns:
        None
    """

    # Declare variables from model_vars
    X_train = model_vars['X_train']
    X_test = model_vars['X_test']
    y_train = model_vars['y_train']

    # Load pre-trained BERT tokenizer and model
    # Using distilbert-base-uncased model as full BERT is too large
    # for my machine
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Padding and tokenisation
    # Truncation is used to ensure the input length is less than the maximum
    # input length of the model
    # For DistilBERT, the maximum input length is 512
    X_train_enc = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='tf')
    X_test_enc = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='tf')

    # Compute class weights - our True label only represents ~33% of the data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    # Set learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    # Callback for learning rate scheduling and early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 2e-5 if epoch < 2 else 1e-5, verbose=1)
    ]

    # Save variables to model_vars
    model_vars['model'] = model
    model_vars['X_train_enc'] = X_train_enc
    model_vars['X_test_enc'] = X_test_enc
    model_vars['class_weights'] = class_weights
    model_vars['optimizer'] = optimizer
    model_vars['callbacks'] = callbacks


def train_model(model_vars: Dict[str, Any]) -> None:
    """
    Train the model.
    BERT works well with larger batch sizes and fewer epochs.
    Epochs are set to 4 and batch size is set to 12 (limited by GPU memory).

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.
    """

    # Declare variables from model_vars
    model = model_vars['model']

    model.compile(
        optimizer=model_vars['optimizer'],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        model_vars['X_train_enc'],
        model_vars['y_train'],
        epochs=4,
        batch_size=12,
        validation_split=0.2,
        class_weight=model_vars['class_weights'],
        callbacks=model_vars['callbacks']
    )

def predict(model_vars: Dict[str, Any]) -> None:
    """
    Perform predictions using the trained model.

    BERT outputs logits, which are converted to probabilities using softmax.
    The probabilities are then converted to binary predictions using a threshold of 0.5.

    Args:
        model_vars (Dict[str, Any]): A dictionary containing the required variables for the model.
    """

    # Declare variables from model_vars
    model = model_vars['model']

    y_pred_nb_proba = model.predict(model_vars['X_test_enc'])

    # Convery logits to probabilities
    y_pred_nb_proba = tf.nn.softmax(y_pred_nb_proba.logits, axis=-1).numpy()  # Convert logits to probabilities
    # Convert to single aray for positive class
    y_pred_nb_proba = y_pred_nb_proba[:, 1]


    print(y_pred_nb_proba)
    # y_pred_nb_proba = y_pred_nb_proba.flatten()
    y_pred_nb = np.where(y_pred_nb_proba > 0.5, 1, 0)
    
    # Save variables to model_vars
    model_vars['y_pred_nb_proba'] = y_pred_nb_proba
    model_vars['y_pred_nb'] = y_pred_nb


if __name__ == '__main__':

    model_vars = {}

    # Parse command line arguments
    MODEL_NAME = parse_args(MODEL_NAME, model_vars)

    run_model_pipeline(MODEL_NAME, fetch_data, declare_model, train_model, predict, model_vars)
