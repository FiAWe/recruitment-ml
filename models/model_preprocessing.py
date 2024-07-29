import pandas as pd
import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder



# Preprocessing
def preprocess_text(text:str) -> str:
    """Preprocess text data.

    Args:
        text (str): input text

    Returns:
        str: preprocessed text
    """

    # Implement text preprocessing steps: lowercase, remove punctuation, etc.
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation ( keep only letters, numbers, "'" and "-")
    text = re.sub(r'[^a-z0-9\'-]', ' ', text)

    # replace '--' with ' -- ' so it will be treated as a separate word
    text = re.sub(r'--', ' -- ', text)
    
    # Replace all whitespace with a single space
    text = re.sub(r'\s+', ' ', text)

    # Remove leading and trailing whitespaces
    text = text.strip()
    
    return text

def generate_all_substrings(text:str, min_length=3, max_length=100) -> list:
    """Generate all possible substrings from a given text.

    Args:
        text (str): input text
        min_length (int): minimum length of the substring
        max_length (int): maximum length of the substring

    Returns:
        list: a list of substrings
    """
    substrings = []
    text_words = text.split()
    text_length = len(text_words)
    for start in range(text_length):
        for end in range(start + min_length, min(start + max_length + 1, text_length + 1)):
            substrings.append(text_words[start:end])
    return substrings

def generate_random_substrings(text:str, num_substrings=10, min_length=10, max_length=100) -> set:
    """Generate random substrings from a given text.

    Args:
        text (str): input text
        num_substrings (int): number of substrings to generate
        min_length (int): minimum length of the substring
        max_length (int): maximum length of the substring

    Returns:
        list: a list of substrings
    """
    substrings = set()
    text_words = text.split()
    text_length = len(text_words)

    if text_length <= min_length:
        return substrings

    max_length = min(max_length, text_length-1)

    # We want to sample shorter substrings more frequently
    # We will use the inverse of the length as the weights
    indices = np.arange(min_length, max_length + 1)
    weights = 1 / indices
    weights /= np.sum(weights) # Normalize the weights

    for _ in range(num_substrings):

        # Choose a random length
        length = np.random.choice(indices, p=weights)
        # Choose a random start position (linear distribution)
        start = np.random.randint(0, text_length - length + 1)
        end = start + length

        new_substring = ' '.join(text_words[start:end])
        substrings.add(new_substring)

    return substrings

def convert_substrings_to_rows(df:pd.DataFrame) -> pd.DataFrame:
    """Convert substrings to rows in a DataFrame.

    Args:
        df (pd.DataFrame): input DataFrame with substrings column

    Returns:
        pd.DataFrame: a new DataFrame with substrings as rows
    """
    # Explode the substrings column and relabel to text
    new_phrases = df.explode('substrings').reset_index(drop=True)
    new_phrases['text'] = new_phrases['substrings']

    # Drop the substrings column
    new_phrases.drop(columns=['substrings'], inplace=True)
    df.drop(columns=['substrings'], inplace=True)

    df = pd.concat([df, new_phrases], ignore_index=True)
    df.drop_duplicates(subset=['text'], inplace=True)

    return df

def get_data(random_state:int=42, generate_substrings:str='none', random_substrings:int=10):
    """Load and preprocess the data.

    Args:
        random_state (int): random seed
        generate_substrings (str): generate all or random substrings. 
            Options: 'all', 'random', 'none'. Default: 'none'
        random_substrings (int): number of random substrings to generate. Default: 10

    Returns:
        X_train: training text data
        X_test: testing text data
        y_train: training labels
        y_test: testing labels
        le: label encoder
    """
    # Load the dataset
    with open('../data/gutenberg-paragraphs.json') as f:
        data = json.load(f)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    df['text'] = df['text'].apply(preprocess_text)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['austen'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['label'],
        test_size=0.2,
        random_state=random_state
    )

    # Generate substrings
    if generate_substrings != 'none':
        X_train, y_train = process_substrings_postsplit(
            generate_substrings,
            X_train, y_train,
            random_state,
            random_substrings
        )

    return X_train, X_test, y_train, y_test, le

def process_substrings_postsplit(generate_substrings:str, X_train:pd.Series, y_train:pd.Series, random_state:int, random_substrings:int):

    # Add the labels back to the data
    X_train = pd.DataFrame(X_train, columns=['text']) # Convert from Series to DataFrame
    X_train['label'] = y_train

    # Generate substrings
    if generate_substrings == 'all':
        X_train['substrings'] = X_train['text'].apply(generate_all_substrings)
    elif generate_substrings == 'random':
        X_train['substrings'] = X_train['text'].apply(generate_random_substrings, num_substrings=random_substrings)
    else:
        raise ValueError('Invalid value for generate_substrings')

    X_train = convert_substrings_to_rows(X_train)

    # Some na values can occur and cause issues with the model
    X_train.dropna(inplace=True)

    # Shuffle the data to mix the substrings with the original text
    # and avoid clumping of the same text
    X_train = X_train.sample(frac=1, random_state=random_state, replace=False).reset_index(drop=True)

    y_train = X_train['label']
    X_train = X_train['text']

    return X_train,y_train
