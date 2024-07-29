import time

import argparse

from model_preprocessing import get_data
from prediction_processing import post_process, save_meta_data

class Timer:
    """
    A simple timer class for measuring the execution time of code blocks.
    """

    def __init__(self):
        self.timings = {}
        self.time_taken = {}

    def start(self, name='main') -> None:
        """
        Start the timer for the specified code block.

        Parameters:
            name (str): The name of the code block. Defaults to 'main'.
        """
        self.timings[f'{name}:start'] = time.perf_counter()

    def end(self, name='main')-> None:
        """
        Stop the timer for the specified code block and calculate the elapsed time.

        Parameters:
            name (str): The name of the code block. Defaults to 'main'.
        """
        self.timings[f'{name}:end'] = time.perf_counter()
        if f'{name}:start' in self.timings:
            self.time_taken[name] =\
                 self.timings[f'{name}:end'] - self.timings[f'{name}:start']

    def __getitem__(self, key:str)-> float:
        """
        Get the elapsed time for the specified code block.

        Parameters:
            key (str): The name of the code block.

        Returns:
            float: The elapsed time in seconds.
        """
        return self.time_taken[key]

def parse_args(MODEL_NAME):
    parser = argparse.ArgumentParser(description='Naive Bayes model')
    parser.add_argument('--generate_substrings', type=str, default='none', const='random', nargs='?', help='Generate random substrings')
    parser.add_argument('--random_substrings', type=int, default=10, help='Number of random substrings to generate')
    args = parser.parse_args()

    # Append model name with 'random' and number of random substrings 
    # if generating random substrings
    if args.generate_substrings == 'random':
        MODEL_NAME += f'__random_{args.random_substrings}'
    return args, MODEL_NAME

def fetch_data(model_vars:dict, args:argparse.Namespace)-> None:
    """
    Fetches the data for training and testing the model and stores it in the model_vars dictionary.

    This function retrieves the training and testing data, along with their corresponding labels,
    and a label encoder. The data is fetched based on the command-line arguments provided.

    Args:
        model_vars (dict): A dictionary to store the fetched data. The dictionary will be updated
                           with the following keys:
                           - 'X_train' (np.array): The training data.
                           - 'X_test' (np.array): The testing data.
                           - 'y_train' (np.array): The training labels.
                           - 'y_test' (np.array): The testing labels.
                           - 'le' (LabelEncoder): The label encoder.
        args (Namespace): An object containing the command-line arguments. It should have the following attributes:
                          - generate_substrings (bool): Whether to generate substrings.
                          - random_substrings (bool): Whether to generate random substrings.

    Returns:
        None
    """

    X_train, X_test, y_train, y_test, le = get_data(
        generate_substrings=args.generate_substrings,
        random_substrings=args.random_substrings
    )

    # Save variables to model_vars
    model_vars['X_train'] = X_train
    model_vars['X_test'] = X_test
    model_vars['y_train'] = y_train
    model_vars['y_test'] = y_test
    model_vars['le'] = le

def run_model_pipeline(
        MODEL_NAME:str,
        fetch_data:callable,
        declare_model:callable,
        train_model:callable,
        predict:callable,
        model_vars:dict,
        args:argparse.Namespace
    ):
    """
    Runs the model pipeline for training and evaluating a machine learning model.

    Args:
        MODEL_NAME (str): The name of the model.
        fetch_data (callable): A function that fetches the data for training and evaluation.
        declare_model (callable): A function that declares the model architecture.
        train_model (callable): A function that trains the model.
        predict (callable): A function that performs predictions using the trained model.
        model_vars (dict): A dictionary containing variables and parameters for the model.
        args (Namespace): A namespace object containing command-line arguments.

    Returns:
        None
    """

    print(f'Model name: {MODEL_NAME}')

    timer = Timer()
    timer.start()

    # Get the data
    timer.start('get_data')
    fetch_data(model_vars, args)
    timer.end('get_data')

    print(f'Data preparation time: {timer["get_data"]:0.4f}')

    # Declare the model
    timer.start('model_declaration')
    declare_model(model_vars)
    timer.end('model_declaration')

    # Fit the model
    timer.start('fit')
    train_model(model_vars)
    timer.end('fit')

    print(f'Training time: {timer["fit"]:0.4f}')

    # Evaluate the model
    timer.start('predict')
    predict(model_vars)
    timer.end('predict')
    print(f'Prediction time: {timer["predict"]:0.4f}')

    # Process results and save figures/data
    post_process(model_vars, model_name=MODEL_NAME)
    timer.end()
    save_meta_data(timer.timings, model_vars, model_name=MODEL_NAME)
