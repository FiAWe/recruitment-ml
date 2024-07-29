import time
import argparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from model_preprocessing import get_data
from prediction_processing import post_process, save_meta_data

MODEL_NAME = 'naive_bayes_ngram'

timings = {}

timings['start'] = time.perf_counter()
timings['model_declaration:start'] = time.perf_counter()
# Pipeline for Naive Bayes with TF-IDF

# Vectorizers:
# CountVectorizer
# HashingVectorizer
# TfidfVectorizer
# TfidfVectorizer with ngram_range
# Word2Vec

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
    ('nb', MultinomialNB())
])
timings['model_declaration:end'] = time.perf_counter()


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

    print(f'Data preparation time: {timings["get_data:end"] - timings["get_data:start"]:0.4f}')

    # Fit the model
    timings['fit:start'] = time.perf_counter()
    pipeline_nb.fit(X_train, y_train)
    timings['fit:end'] = time.perf_counter()

    print(f'Training time: {timings["fit:end"] - timings["fit:start"]:0.4f}')

    # Evaluate the model
    timings['predict:start'] = time.perf_counter()
    y_pred_nb = pipeline_nb.predict(X_test)
    timings['predict:end'] = time.perf_counter()
    print(f'Prediction time: {timings["predict:end"] - timings["predict:start"]:0.4f}')
    y_pred_nb_proba = pipeline_nb.predict_proba(X_test)[:, 1]

    timings['end'] = time.perf_counter()

    post_process(X_test, y_test, y_pred_nb, y_pred_nb_proba, le, model_name=MODEL_NAME)

    data_size = {
        'train': len(X_train),
        'test': len(X_test)
    }

    save_meta_data(timings, data_size, model_name=MODEL_NAME)
