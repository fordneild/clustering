""" Main file. This is the starting point for your code execution.

You shouldn't need to change anything in this code.
"""

import os
import argparse as ap
import pickle
import numpy as np

import models
from data import load_data


def get_args():
    p = ap.ArgumentParser(description="This is the main test harness for your models.")

    # Meta arguments
    p.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                    help="Operating mode: train or test.")
    p.add_argument("--train-data", type=str, help="Training data file")
    p.add_argument("--test-data", type=str, help="Test data file")
    p.add_argument("--model-file", type=str, required=True,
                    help="Where to store and load the model parameters")
    p.add_argument("--predictions-file", type=str,
                    help="Where to dump predictions")
    p.add_argument("--algorithm", type=str,
                   choices=['lambda_means', 'stochastic_k_means'],
                    help="The type of model to use.")

    # Model Hyperparameters
    p.add_argument("--cluster-lambda", type=float,
                        help="The value of lambda in lambda-means", default=0.0)
    p.add_argument("--clustering-training-iterations", type=int,
                        help="The number of clustering iterations", default=10)
    p.add_argument("--number-of-clusters", type=int,
                        help="The number of clusters (K) to be used.", default=3)
    return p.parse_args()


def check_args(args):
    mandatory_args = {'mode', 'model_file', 'test_data', 'train_data', 'algorithm',
                      'predictions_file', 'cluster_lambda', 'clustering_training_iterations',
                      'number_of_clusters'}
    if not mandatory_args.issubset(set(dir(args))):
        raise Exception("You're missing essential arguments!"
                         "We need these to run your code.")

    if args.mode.lower() == "train":
        if args.algorithm is None:
            raise Exception("--algorithm should be specified during training")
        if args.model_file is None:
            raise Exception("--model-file should be specified during training")
        if args.train_data is None:
            raise Exception("--train-data should be specified during training")
        elif not os.path.exists(args.train_data):
            raise Exception("data file specified by --train-data does not exist.")
    elif args.mode.lower() == "test":
        if args.predictions_file is None:
            raise Exception("--predictions-file should be specified during testing")
        if not os.path.exists(args.model_file):
            raise Exception("model file specified by --model-file does not exist.")
        if args.test_data is None:
            raise Exception("--test-data should be specified during testing")
        elif not os.path.exists(args.test_data):
            raise Exception("data file specified by --test-data does not exist.")
    else:
        raise Exception("invalid mode")

def train(args):
    """ 
    Fit a model's parameters given the parameters specified in args.
    """
    # Load the training data.
    X, _ = load_data(args.train_data)

    # build the appropriate model
    if args.algorithm.lower() == 'lambda_means':
        model = models.LambdaMeans(nfeatures=X.shape[1], lambda0=args.cluster_lambda)
    elif args.algorithm.lower() == 'stochastic_k_means':
        model = models.StochasticKMeans(nfeatures=X.shape[1], num_clusters=args.number_of_clusters)
    else:
        raise Exception('The model given by --model is not yet supported.')

    # Run the training loop
    model.fit(X=X, iterations=args.clustering_training_iterations)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))


def test(args):
    """ 
    Make predictions over the input test dataset, and store the predictions.
    """
    # load dataset and model
    X, _ = load_data(args.test_data)
    model = pickle.load(open(args.model_file, 'rb'))

    # predict labels for dataset
    preds = model.predict(X)
    
    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%d')


if __name__ == "__main__":
    args = get_args()
    check_args(args)

    if args.mode.lower() == 'train':
        train(args)
    elif args.mode.lower() == 'test':
        test(args)
    else:
        raise Exception("Mode given by --mode is unrecognized.")
