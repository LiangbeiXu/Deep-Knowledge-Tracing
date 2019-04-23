# compare the two datasets
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import compress
data_original_file = "data/ASSISTments_skill_builder_data.csv"
data_download_file = "data/skill_builder_data.csv"
data_correct_file = "data/skill_builder_data_corrected_withskills.csv"


def PreprocessAssistment(file_path):
    data = pd.read_csv(file_path,dtype={'skill_name':np.str, 'order_id':np.int, \
        'problem_id':np.int, 'user_id':np.int, 'correct':np.int}, usecols =['order_id', 'skill_id', 'user_id', 'correct', 'skill_name', 'problem_id'])
    data.drop(axis=1,columns='skill_name', inplace=True)
    print('# of records: %d.' %(data.shape[0]))
    data.dropna(inplace=True)
    print('After dropping NaN rows, # of records: %d.' %(data.shape[0]))
    data.sort_values('order_id',ascending=True, inplace=True)
    return data

original = PreprocessAssistment(data_original_file)
download = PreprocessAssistment(data_download_file)

for i in range(original.shape[1]):
    for j in range(original.shape[0]):
        issame = (original.iloc[j,i] == download.iloc[j,i])
        if not issame:
            print('%d, %d'%(original.columns[i], j))


correct  = PreprocessAssistment(data_correct_file)

