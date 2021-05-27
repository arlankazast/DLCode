import numpy as np
import random
import tensorflow.compat.v1 as tf

def random_seed(seed_value):
    np.random.seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    tf.set_random_seed(seed_value)


random_seed(42)
# random.seed(42)

from classifier import TextClassifier
from config import Config

from dataset.vector_data_loader import VectorDataLoader
from utils import Utils

import warnings
warnings.filterwarnings("ignore")

def load_datasets(config):
    datasets={}
    if config.data.format=="vectors":
        for file in config.data.vectors.path_to_vector_file:
            dataset=VectorDataLoader(config,file)
            datasets[file]=dataset

    return datasets


import sys
if __name__=="__main__":
    model_name=sys.argv[1]

    config = Config(config_file="config/vectors_up.json")
    #load model names from configuration
    # model_names=config.model_names
    #load dataset
    datasets=load_datasets(config)
    #select and load evaluator
    evaluator=Utils.get_evaluator(config)
    #iterate over all datasets to generate results
    for dataset_name, dataset in datasets.items():

        #initialize dataset
        print("Initializing Dataset: "+dataset_name)
        dataset.init()

        #initialize classifier for the dataset

        #iterate over all models to perform task
        # for model_name in model_names:
        classifier = TextClassifier(config, dataset, evaluator)

        try:
            print(
                "----------------------------------------------------------------------------------------------------------")
            print("Dataset: " + dataset_name + "    Model: " + model_name)
            print(
                "----------------------------------------------------------------------------------------------------------")

            # perform task on the dataset mentioned in the configuration
            results = classifier.train_test(model_name)

            # save results in a file if user wants
            if config.save_results:
                results = np.average(np.asarray(results).reshape(-1, (np.asarray(results).shape[0])), axis=0)
                evaluator.results_log(dataset_name, model_name, *results,specificity=1)
        except Exception as e:
            print(e)