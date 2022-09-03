"""
Requirments:
- Run on IBM virtual machine, detail see VM_setup.md
- Environment: openml, pyspark, pmlb
"""

import srom
import openml as oml
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import dill
import pickle

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from pmlb import fetch_data, classification_dataset_names
from srom.auto.auto_classification import AutoClassification

import warnings
warnings.filterwarnings('ignore')

test_size = 0.2 # train:test=8:2

######################### Function Part #########################
def write_first_pickle(i):
    """
    Store the result for i-th dataset to the pickle file
    """
    tensor, est_info = matrix_3d(i, i+1)
    print('Done run matrix_3d: dataset', i)

    filename1, filename2 = 'tensor.pickle', 'est_info.pickle'
    pickle.dump(tensor, open(filename1, "wb"))
    pickle.dump(est_info, open(filename2, "wb"))

    print('Done save new data: dataset', i)
    print('===========================================================')


def get_class_matrix(all_estimators, X, y):
    """
    Return a 2D matrix and a list of detail information of each estimator/pipeline
    2D matrix:
    each row is an estimator, each column an dataset instance,
    each entry 1/0 (FALSE/TRUE classification)
    :param all_estimators: a list of all estimators (Pipeline object)
    :param (X, y): dataset
    :return: a binary 2D matrix and a list of pipeline detail info
    """
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Construct the matrix
    res_matrix = np.zeros(y_test.shape, dtype=int)
    est_lst = []
    for estimator in all_estimators:
        if estimator.get_params()['steps'][0][0] == 'xgbclassifier':
            continue
        if estimator is not None:
            try:
                curr_model = estimator.fit(X_train, y_train)
            except ValueError:
                continue
            else:
                try:
                    y_pred = estimator.predict(X_test)
                except ValueError:
                    continue
                else:
                    curr_diff = y_pred - y_test
                    curr_row = np.where(curr_diff != 0, 1, 0)
                    est_lst.append(estimator)
                    # stack the row
                    res_matrix = np.vstack([res_matrix, curr_row])
    
    # escape the first row
    return res_matrix[1:, :], est_lst


def matrix_3d(i, j):
    """
    Return 2 dictionary:
    tensor = {'dataset_name': 2D matrix return from get_class_matrix}
    all_est_info = {'dataset_name': est_info list return from get_class_matrix}
    :param i: start dataset index from pmlb classification_dataset_names
    :param j: end dataset index from pmlb classification_dataset_names
    :return: 2 dict {str: np.array} and {str: list[Pipeline()]}
    """
    tensor, all_est_info = {}, {}
    for n in range(i, j):
        # fetch dataset
        dataset_name = classification_dataset_names[n]
        X, y = fetch_data(dataset_name, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # create auto-classification object, fit the models
        ac = AutoClassification(level='comprehensive', 
                        scoring='accuracy', 
                        cv = 5,
                        total_execution_time=10, 
                        save_prefix='auto_classification_output_')
        ac.automate(X,y)
        ac.fit(X_train, y_train)
        
        # get best estimators' information
        ac.export_pipeline_exploration_info()
        with open(ac.dill_filename, 'rb') as file:
            best_path_info = dill.load(file)
        all_estimators = []
        for est_info in best_path_info['best_path']:
            all_estimators.append(est_info['best_estimator'])
        
        # get the 2D matrix of current dataset
        final_matrix, est_lst = get_class_matrix(all_estimators, X, y)
        
        # construct the tensor and keep track of the info
        tensor[dataset_name] = final_matrix
        all_est_info[dataset_name] = est_lst
    
    return tensor, all_est_info


def run_and_store(i, j):
    """
    Run and store tensor and est_info for each dataset
    """
    for n in range(i, j):
        # readout data from pickle file
        filename1, filename2 = 'tensor.pickle', 'est_info.pickle'
        tensor_data, est_info_data = pickle.load(open(filename1, "rb")), pickle.load(open(filename2, "rb"))

        print('Done readout data: dataset', n)

        # get results for current dataset
        tensor, est_info = matrix_3d(n, n+1)
        print('Done run matrix_3d: dataset', n)

        # concatenate data
        new_tensor = tensor_data | tensor
        new_est_info = est_info_data | est_info
        print('Done concatenate data: dataset', n)

        # write to pickle file
        pickle.dump(new_tensor, open(filename1, "wb"))
        pickle.dump(new_est_info, open(filename2, "wb"))

        print('Done save new data: dataset', n)
        print('===========================================================')


#########################################################################

write_first_pickle(0)
run_and_store(1, len(classification_dataset_names))

#########################################################################

