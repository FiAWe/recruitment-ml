#!/bin/bash

random_string_counts=(5 10 50)

# Naive Bayes
python3 naive_bayes.py
for count in "${random_string_counts[@]}"; do
    python3 naive_bayes.py --generate_substrings --random_substrings "$count"
done

# Naive Bayes - ngram vectorizer
python3 naive_bayes_ngram.py
for count in "${random_string_counts[@]}"; do
    python3 naive_bayes_ngram.py --generate_substrings --random_substrings "$count"
done

# Naive Bayes - word2vec vectorizer
python3 naive_bayes_word2vec.py
for count in "${random_string_counts[@]}"; do
    python3 naive_bayes_word2vec.py --generate_substrings --random_substrings "$count"
done

# svm
python3 svm.py
for count in "${random_string_counts[@]}"; do
    python3 svm.py --generate_substrings --random_substrings "$count"
done

# Reducing the number of random substrings for LSTM and BERT
random_string_counts=(5 10)

# lstm
python3 lstm.py
for count in "${random_string_counts[@]}"; do
    python3 lstm.py --generate_substrings --random_substrings "$count"
done

# bert
python3 bert.py
for count in "${random_string_counts[@]}"; do
    python3 bert.py --generate_substrings --random_substrings "$count"
done