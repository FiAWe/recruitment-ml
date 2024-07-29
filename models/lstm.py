import time
import argparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from sklearn.preprocessing import LabelEncoder
import numpy as np

from model_preprocessing import get_data
from prediction_processing import post_process, save_meta_data

MODEL_NAME = 'LSTM'

timings = {}

timings['start'] = time.perf_counter()




if __name__ == '__main__':

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

    timings['model_declaration:start'] = time.perf_counter()
    # Tokenisation
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Padding
    max_len = 300
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    # LSTM Model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_len))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    timings['model_declaration:end'] = time.perf_counter()

    print(f'Data preparation time: {timings["get_data:end"] - timings["get_data:start"]:0.4f}')

    # Fit the model
    timings['fit:start'] = time.perf_counter()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=10, batch_size=64)
    timings['fit:end'] = time.perf_counter()

    print(f'Training time: {timings["fit:end"] - timings["fit:start"]:0.4f}')

    # Evaluate the model
    timings['predict:start'] = time.perf_counter()
    y_pred_nb_proba = model.predict(X_test_pad)
    y_pred_nb_proba = y_pred_nb_proba.flatten()
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
