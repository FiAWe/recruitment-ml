import time
import argparse

from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import numpy as np

from model_preprocessing import get_data
from prediction_processing import post_process, save_meta_data

from sklearn.utils.class_weight import compute_class_weight

MODEL_NAME = 'BERT'



timings = {}

if __name__ == '__main__':

    timings['start'] = time.perf_counter()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Naive Bayes model')
    parser.add_argument('--generate_substrings', type=str, default='none', const='random', nargs='?', help='Generate random substrings')
    parser.add_argument('--random_substrings', type=int, default=10, help='Number of random substrings to generate')
    args = parser.parse_args()

    # Append model name with 'random' and number of random substrings 
    # if generating random substrings
    if args.generate_substrings == 'random':
        MODEL_NAME += f'__random_{args.random_substrings}'

    print(f'Model name: {MODEL_NAME}')

    # Get the data
    timings['get_data:start'] = time.perf_counter()
    X_train, X_test, y_train, y_test, le = get_data(
        generate_substrings=args.generate_substrings,
        random_substrings=args.random_substrings
        )
    timings['get_data:end'] = time.perf_counter()

    print(f'Data preparation time: {timings["get_data:end"] - timings["get_data:start"]:0.4f}')

    timings['model_declaration:start'] = time.perf_counter()

    # Load pre-trained BERT tokenizer and model
    # Using distilbert-base-uncased model as full BERT is too large
    # for my machine
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)


    # Padding and tokenisation
    X_train_enc = tokenizer(list(X_train), padding=True, truncation=True, return_tensors='tf')
    X_test_enc = tokenizer(list(X_test), padding=True, truncation=True, return_tensors='tf')

    # # For code testing just take 1/20 of the data
    # X_train_enc = {key: value[:len(value)//20] for key, value in X_train_enc.items()}
    # y_train = y_train[:len(y_train)//20]

    timings['model_declaration:end'] = time.perf_counter()

    # Fit the model
    timings['fit:start'] = time.perf_counter()

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}
    # class_weights = tf.constant(list(class_weights.values()))


    # Set learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Callback for learning rate scheduling and early stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 2e-5 if epoch < 2 else 1e-5, verbose=1)
    ]

    model.fit(
        X_train_enc,
        y_train,
        epochs=4,
        batch_size=12,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=callbacks
    )

    timings['fit:end'] = time.perf_counter()

    print(f'Training time: {timings["fit:end"] - timings["fit:start"]:0.4f}')

    # Evaluate the model
    timings['predict:start'] = time.perf_counter()
    y_pred_nb_proba = model.predict(X_test_enc)

    # Convery logits to probabilities
    y_pred_nb_proba = tf.nn.softmax(y_pred_nb_proba.logits, axis=-1).numpy()  # Convert logits to probabilities
    # Convert to single aray for positive class
    y_pred_nb_proba = y_pred_nb_proba[:, 1]


    print(y_pred_nb_proba)
    # y_pred_nb_proba = y_pred_nb_proba.flatten()

    timings['predict:end'] = time.perf_counter()
    print(f'Prediction time: {timings["predict:end"] - timings["predict:start"]:0.4f}')
    y_pred_nb = np.where(y_pred_nb_proba > 0.5, 1, 0)

    timings['end'] = time.perf_counter()

    post_process(X_test, y_test, y_pred_nb, y_pred_nb_proba, le, model_name=MODEL_NAME)

    data_size = {
        'train': len(X_train),
        'test': len(X_test)
    }

    save_meta_data(timings, data_size, model_name=MODEL_NAME)
