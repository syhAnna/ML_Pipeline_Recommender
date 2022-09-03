import srom
import openml as oml
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import dill
import pickle
import os

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from pmlb import fetch_data, classification_dataset_names
from srom.auto.auto_classification import AutoClassification

import warnings
warnings.filterwarnings('ignore')

test_size = 0.2

###########################################################
# Construct 3D (datasetID, pipelineID, instanceID) tensor #
###########################################################
def sort_by_key(filename1, filename2):
    # readout the data
    tensor_data, est_info_data = pickle.load(open(filename1, "rb")), pickle.load(open(filename2, "rb"))
    # sort alphabetically
    sorted_items1 = tensor_data.items()
    new_items1 = sorted(sorted_items1)
    new_tensor_data = {key: val for (key, val) in new_items1}
    sorted_items2 = est_info_data.items()
    new_items2 = sorted(sorted_items2)
    new_est_info_data = {key: val for (key, val) in new_items2}
    # rewrite into the pickle file
    pickle.dump(new_tensor_data, open(filename1, "wb"))
    pickle.dump(new_est_info_data, open(filename2, "wb"))
    return new_tensor_data


def get_tensor3d(filename1, filename2):
    # readout the data
    tensor_data = pickle.load(open(filename1, "rb"))
    max_row, max_col = 0, 0
    for key, val in tensor_data.items():
        curr_row, curr_col = val.shape
        max_row, max_col = max(max_row, curr_row), max(max_col, curr_col)
    # construct a 3D-tensor
    tensor_3d = np.empty((len(tensor_data), max_row, max_col)) # (dataset, pipeline, data_instance)
    tensor_3d[:] = np.NaN
    layer = 0
    new_tensor_data = sort_by_key(filename1, filename2)
    for key, val in new_tensor_data.items():
        row, col = val.shape
        tensor_3d[layer][:row, :col] = val
        layer += 1
    # write result into pickle file
    root = os.getcwd() + '/data/multiple_datasets/'
    filename = 'tensor3d.pickle'
    pickle.dump(tensor_3d, open(root+filename, "wb"))
    return tensor_3d


#####################################################
# Construct 3D (pipeline, Dataset, Accuracy) tensor #
#####################################################
def accuarcy_1d(matrix2d):
    # input a 2d matrix (pipelines[index], instances)
    # access specific pipelines: est_info_data['dataset'][index]
    pipelines, instances = matrix2d.shape
    pipeline_acc = np.empty((pipelines, 1))
    pipeline_acc[:] = np.NaN
    for i in range(pipelines):
        curr_row = matrix2d[i][:]
        count, wrong = 0, 0
        while count < len(curr_row) and not np.isnan(curr_row[count]):
            wrong += curr_row[count]
            count += 1
        try:
            curr = (count - wrong) / count
        except ZeroDivisionError:
            continue
        else:
            pipeline_acc[i][0] = curr
    return pipeline_acc


def construct_2d(tensor3d):
    curr = accuarcy_1d(tensor3d[0])
    for i in range(1, len(tensor3d)):
        nxt = accuarcy_1d(tensor3d[i])
        curr = np.concatenate((curr,nxt), axis=1)
    return curr


def overall_construct(filename1, filename2):
    # readout 3d tensor data
    data, tensor_data = pickle.load(open(filename1, "rb")), pickle.load(open(filename2, "rb"))
    dataset_id = list(tensor_data.keys())
    res = construct_2d(data)
    # write result to pickle file
    root = os.getcwd() + '/data/multiple_datasets/'
    f1, f2 = 'ppl_dataset2d.pickle', 'datasets_mapping.pickle'
    pickle.dump(res, open(root+f1, "wb"))
    pickle.dump(dataset_id, open(root+f2, "wb"))
    return res, dataset_id


def get_pipeline_detail(pipeline_id, dataset_id):
    """
    Input the pipeline_id (0, 312) total 313
    dataset_id (0, 120) total 121
    (pipeline_id, dataset_id) is 2d accuracy matrix id (313, 121)
    Output the correspond dataset's pipeline detail
    """
    root = os.getcwd() + '/data/multiple_datasets/'
    filename1, filename2, filename3 = 'datasets_mapping.pickle', 'est_info.pickle', 'ppl_dataset2d.pickle'
    data_map, est_info_data, acc_data = pickle.load(open(root+filename1, "rb")), pickle.load(open(root+filename2, "rb")), pickle.load(open(root+filename3, "rb"))
    # get name of the datasets
    datasets_name = data_map[dataset_id]
    ppl_detail = est_info_data[datasets_name][pipeline_id]
    accuracy = acc_data[pipeline_id, dataset_id]
    print("Dataset: ", datasets_name)
    print("Pipeline detail: ", ppl_detail)
    print("Has accuracy: ", accuracy)
    return ppl_detail


#########################################################################
root = os.getcwd() + '/data/multiple_datasets/'
filename1, filename2 = 'tensor.pickle', 'est_info.pickle'
# construct the 3d tensor
tensor_3d = get_tensor3d(root+filename1, root+filename2)
#########################################################################
filename1, filename2 = 'tensor3d.pickle', 'tensor.pickle'
res, dataset_id = overall_construct(filename1, filename2)
#########################################################################

