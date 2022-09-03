import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import re
from collections import defaultdict, OrderedDict

############################## Load raw data ##############################

input_root = os.getcwd() + '/data/multiple_datasets/'
filename1 = 'datasets_mapping.pickle'
filename2 = 'pplID_mapping_df.csv'
filename3 = 'tensor.pickle'

dataset_names = pickle.load(open(input_root + filename1, "rb"))
est_info_df = pd.read_csv(input_root + filename2)
tensor_dict = pickle.load(open(input_root + filename3, "rb"))

curr_tensor = tensor_dict[dataset_names[0]]
row, col = curr_tensor.shape
curr_row_matrix = curr_tensor[0, :]

############################ Helper functions ############################

def find_next(begin, lst):
  """
  Return the smallest value greater than begin in lst
  """
  res = [val for val in lst if val > begin]
  return min(res)


def get_name(input_str):
  """
  Given a input_str, return a list of all estimator names inside ''
    
  input_str example:
  "Pipeline(steps=[('discretizer', KBinsDiscretizer()),
              ('mlpclassifier',
                MLPClassifier(alpha=0.1, hidden_layer_sizes=(5, 10),
                              random_state=42, solver='lbfgs'))])"
    
  return a list of string:
  ['discretizer', 'mlpclassifier']
  """
  indices_begin = [index for index in range(len(input_str)) if input_str.startswith("('", index)]
  indices_end = [index for index in range(len(input_str)) if input_str.startswith("',", index)]
  ppl_names = []
  for begin in indices_begin:
    end = find_next(begin, indices_end)
    ppl_names.append(input_str[begin+2:end])
  return ppl_names


def str_to_np_array(df):
  """
  Convert a string of numpy to numpy
  """
  result = df['row_matrix'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
  return result


#################################################
# Create a dataframe .csv file for each dataset #
#################################################
def create_one_file(dataset_id, dataset_name, filename):
  """
  Create a dataframe .csv file for dataset_name:
   pipeline_names | row_matrix | accuracy
  Input: 
  dataset_id: id
  dataset_name: str
  filename: str, the .csv filename that to be created
  """
  df = pd.DataFrame(columns=('pipeline_names', 'row_matrix', 'accuracy'))
  curr_tensor = tensor_dict[dataset_name]
  pipelines = est_info_df.iloc[dataset_id, :]

  ppl_id = 0
  for ppl in pipelines:
    if str(ppl) != 'nan':
      curr_ppl = get_name(str(ppl))
      curr_row_matrix = curr_tensor[ppl_id, :]
      curr_acc = (len(curr_row_matrix) - sum(curr_row_matrix)) / len(curr_row_matrix)
      df.loc[ppl_id] = [curr_ppl, curr_row_matrix, curr_acc]
      ppl_id += 1
  # write df into .csv file
  df.to_csv(filename, encoding='utf-8', index=False)


def create_multiple_files():
  """
  For each dataset, create a dataframe .csv file:
    pipeline_name | row_matrix | accuracy
  """
  # create the out_files directory
  new_dir = 'output_files'
  parent_dir = os.getcwd() + '/data/'
  path = os.path.join(parent_dir, new_dir)
  os.mkdir(path)

  output_root = os.getcwd() + '/data/output_files/'
  for i in range(len(dataset_names)):
    filename = str(i)+'_'+dataset_names[i]+'.csv'
    create_one_file(i, dataset_names[i], output_root+filename)


####################### Refine the dataframe files #######################
# For each dataset (already has an dataframe .csv file from above output):
# *   sort the dataframe by descending of accuracy
# *   keep the unique pipeline_name with highest accuracy
##########################################################################
def refine_one_df(loaded_df):
  """
  Input a dataframe: pipeline_name | row_matrix | accuracy
  Output a dataframe with refine process:
  - sort the dataframe by descending of accuracy
  - keep the unique pipeline_name with highest accuracy
  """
  # sort the dataframe by descending of accuracy
  sorted_df = loaded_df.sort_values(by=['accuracy'], ascending=False)
  # keep the unique pipeline_name with highest accuracy
  refined_df = sorted_df.drop_duplicates(subset=['pipeline_names'], keep='first')
  return refined_df


def refine_multiple_df():
  # create refined_output directory for refined output
  new_dir = 'refined_output_files'
  parent_dir = os.getcwd() + '/data/'
  path = os.path.join(parent_dir, new_dir)
  os.mkdir(path)

  input_root = os.getcwd() + '/data/output_files/'
  output_root = os.getcwd() + '/data/refined_output_files/'
  for filename in os.listdir(input_root):
    filepath = input_root + filename
    loaded_df = pd.read_csv(filepath)
    refined_df = refine_one_df(loaded_df)
    newfile = output_root + 'refined_' + filename
    refined_df.to_csv(newfile, encoding='utf-8', index=False)


#######################################################################################
# Choose topK=5 pipelines that cover most instances                                   #
# Find a set of topK=5 pipelines that can cover most of instances as much as possible # 
# (dataframe sorted in descending accuracy order and with unique pipeline_names)      #
#######################################################################################
topK = 5

def get_topK_ppl(df, topK):
  """
  Input a dataframe, topK
  Output correspond dataframe that chosen topK pipelines
  """
  # first k-1 chosen by accuracy
  topK_df = df.iloc[:(topK-1) , :].copy()
  remain_df = df.iloc[(topK-1): , :].copy()

  # next choose by coverage
  top_arr = str_to_np_array(topK_df)
  res = np.ones(top_arr[0].shape)
  for arr in top_arr:
    res = np.multiply(res, arr)

  # find one in remain part
  curr_product, curr_id = sum(res), 0
  arr_list = list(str_to_np_array(remain_df))
  for i in range(len(arr_list)):
    curr_arr = np.multiply(res, arr_list[i])
    if sum(curr_arr) < curr_product:
      curr_product, curr_id = sum(curr_arr), i

  new_row = remain_df.iloc[i, :]
  topK_df = topK_df.append(new_row)
  return topK_df


def get_all_topK(topK):
  # create output directory
  new_dir = 'topk_pipelines_files'
  parent_dir = os.getcwd() + '/data/'
  path = os.path.join(parent_dir, new_dir)
  os.mkdir(path)

  input_root = os.getcwd() + '/data/refined_output_files/'
  output_root = os.getcwd() + '/data/topk_pipelines_files/'
  file_list=os.listdir(input_root)
  for input_file in file_list:
    output_filename = input_file.split('refined_')[1]
    output_filename = 'topK_' + output_filename
    
    input_df = pd.read_csv(input_root + input_file)
    output_df = get_topK_ppl(input_df, topK)
    output_df.to_csv(output_root + output_filename, encoding='utf-8', index=False)


#################################################################
# Function to construct DataFrame (Dataset, Pipeline, Accuracy) #
#################################################################
def construct_df():
  input_root = os.getcwd() + '/data/refined_output_files/'
  file_list=os.listdir(input_root)
  final_df = pd.DataFrame(columns=('dataset_name', 'pipeline_name', 'accuracy'))

  curr_idx = 0
  for filename in file_list:
    filepath = input_root + filename
    dataset_name = filename.split('refined_')[1].split('.csv')[0]
    curr_df = pd.read_csv(filepath)
    num_ppl = curr_df.shape[0]
    for ppl in range(num_ppl):
      final_df.loc[curr_idx] = [dataset_name, curr_df['pipeline_names'][ppl], curr_df['accuracy'][ppl]]
      curr_idx += 1
  
  root = os.getcwd() + '/data/multiple_datasets/'
  filename = 'data_ppl_acc_finalDF.csv'
  final_df.dropna()
  final_df.to_csv(filename, encoding='utf-8', index=False)


#####################################################################
create_multiple_files()
refine_multiple_df()
#####################################################################
construct_df()
#####################################################################