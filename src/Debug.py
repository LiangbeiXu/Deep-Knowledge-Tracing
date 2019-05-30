#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:31:56 2019

@author: lxu
"""

import sys
sys.path.insert(0, '/home/lxu/Documents/Probabilistic-Matrix-Factorization')
from IRT import IRT
from IRT2 import IRT2
from sklearn.model_selection import train_test_split
from Utils import *
from StudentModel import DKTModel, DataGenerator
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
# dataset = "~/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills.csv" # Dataset path
# dataset = "~/Documents/StudentLearningProcess/2015_100_skill_builders_main_problems.csv" # Dataset path
dataset_file = '~/Documents/Assistment09-problem-single_skill.csv'
# dataset = "data/skill_builder_data.csv"
best_model_file = "saved_models/ASSISTments.best.model.weights.hdf5" # File to save the model.
# best_model_file = "logs/8/saved_models/ASSISTments.best.model.weights.hdf5" # File to save the model.
train_log = "logs/dktmodel.train.log" # File to save the training log.
eval_log = "logs/dktmodel.eval.log" # File to save the testing log.
optimizer = "adagrad" # Optimizer to use
lstm_units = 250 # Number of LSTM units
batch_size = 40 # Batch size
epochs = 10 # Number of epochs to train
dropout_rate = 0.5 # Dropout rate
verbose = 1 # Verbose = {0,1,2}
testing_rate = 0.2 # Portion of data to be used for testing
validation_rate = 0.2 # Portion of training data to be used for validation
embedding_size = 16 # prob and user embedding dimension




def split_data( data, insample, mode):
    if insample:
        if mode == 'Searching':
            train, test = train_test_split(data, test_size=0.2)
            train, validation = train_test_split(train, test_size = 0.1)
        elif mode == 'Testing':
            train, validation = train_test_split(data, test_size = 0.2)
    else:
        validation_size = 20000
        test_size = 60000
        if mode == 'Searching':
            test = data[-int(test_size):,:]
            train = data[0:-int(test_size+validation_size),:]
            validation = data[-int(test_size+validation_size):-int(test_size),:]
        elif mode == 'Testing':
            validation = data[-int(test_size):,:]
            train = data[0:-int(test_size),:]
    return train, validation

data = read_file_for_mf(dataset_file)
# print(data)
# print(data.values[0:10,:])
stats = {'num_entries': data.shape[0], 'num_users': len(np.unique(data['user_id'])), 'num_skills': len(np.unique(data['skill_id'])),
                             'num_probs': len(np.unique(data['problem_id']))}
print(stats)

insample = False
if insample:
    test_size = 0.2
    randp = np.random.uniform(0, 1.0, size=data.shape[0])
    randp[randp > 1-test_size] = 1
    randp[randp <= 1-test_size] = 0

else:
    test_size = 40000
    randp = np.random.uniform(0, 1.0, size=data.shape[0])
    randp[0:-int(test_size)] = 0
    randp[-int(test_size):] = 1
randp = randp.astype(int)
data.insert(data.shape[1], "test", randp, True)
# print(data)

# datanp = data.values
# train, validation = split_data(data=datanp, insample=True, mode='Testing')
# train_data = pd.DataFrame(data=train[:,0:3], columns = ['user_id', 'problem_id', 'correct'])
# test_data = pd.DataFrame(data=validation[:,0:3], columns = ['user_id', 'problem_id', 'correct'])

test_data = data.loc[data['test']==1]
train_data = data.loc[data['test']==0]
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print('train number, test number', train_data.shape, test_data.shape)
# print(train_data, test_data)
# dataset, stats = read_file_with_skill_prob_user(train_data)
num_skills = stats['num_skills']
num_users = stats['num_users']
num_probs = stats['num_probs']

train_data_LSTM = generate_user_data(train_data)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_with_skill_prob_user_flag(train_data_LSTM, 0, 0)




# Create generators for training/testing/validation
train_gen = DataGenerator(X_train, y_train, num_skills, num_users, num_probs, batch_size)

# generate the validation and test
test_data_LSTM = generate_user_data(data)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_with_skill_prob_user_flag(test_data_LSTM, 0, 0)
val_gen = DataGenerator(X_train, y_train, num_skills, num_users, num_probs, batch_size)
test_gen = val_gen

#val_gen = DataGenerator(X_val, y_val, num_skills, num_users, num_probs, batch_size)
#test_gen = DataGenerator(X_test, y_test, num_skills, num_users, num_probs, batch_size)

# X, y = test_gen.next_batch()
print("======== Data Summary ========")
print("Data size: %d" % data.shape[0])
print("Training data size: %d" % len(X_train))
print("Validation data size: %d" % len(X_val))
print("Testing data size: %d" % len(X_test))
print("Number of skills: %d" % num_skills)
print("==============================")


if 0:
    fm = IRT(epsilon=4, _lambda=0.1, momentum=0.8, maxepoch=40, num_batches=300, batch_size=1000,\
                 problem=True, multi_skills=False, user_skill=False, user_prob=False, PFA=False, MF=True, \
                 num_feat=embedding_size, MF_skill=False, user=True, skill_dyn_embeddding=False, skill=False, global_bias=False)
if 1:
    fm = IRT2(epsilon=4, _lambda=0.1, momentum=0.8, maxepoch=40, num_batches=300, batch_size=1000,
                        problem=True, MF_prob=True, num_feat=embedding_size, user=True,  global_bias=False,
                        problem_dyn_embedding=False, patience=5)

fm.fit(train_data, test_data, num_users, num_probs, num_skills)

# Create model
student_model = DKTModel(num_skills=num_skills,
                         num_probs = num_probs,
                         num_users = num_users,
                         embedding_size = embedding_size,
                      num_features=num_skills * 2,
                      optimizer=optimizer,
                      hidden_units=lstm_units,
                      batch_size=batch_size,
                      dropout_rate=dropout_rate)


student_model.model.summary()



if 0:
    print('embeddings from MF:')
    print('item_bias', fm.beta_prob[0:10])
    print('user_bias', fm.beta_user[0:10])
    print('item_embedding', fm.w_prob[0:10,:])
    print('user_embedding', fm.w_user[0:10,:])
item_bias_ones = student_model.model.get_layer('prob_bias').get_weights()
for idx, ele in enumerate(item_bias_ones[0]):
    item_bias_ones[0][idx] = fm.beta_prob[idx]

user_bias_ones = student_model.model.get_layer('user_bias').get_weights()
for idx, ele in enumerate(user_bias_ones[0]):
    user_bias_ones[0][idx] = fm.beta_user[idx]

item_embedding_ones = student_model.model.get_layer('prob_embedding').get_weights()
for idx, ele in enumerate(item_embedding_ones[0]):
    item_embedding_ones[0][idx] = fm.w_prob[idx,:]

user_embedding_ones = student_model.model.get_layer('user_embedding').get_weights()
for idx, ele in enumerate(user_embedding_ones[0]):
    user_embedding_ones[0][idx] = fm.w_user[idx,:]

student_model.model.get_layer('prob_bias').set_weights(item_bias_ones)
student_model.model.get_layer('user_bias').set_weights(user_bias_ones)
student_model.model.get_layer('prob_embedding').set_weights(item_embedding_ones)
student_model.model.get_layer('user_embedding').set_weights(user_embedding_ones)

if 1:
    student_model.model.get_layer('prob_embedding').trainable = False
    student_model.model.get_layer('user_embedding').trainable = False
    student_model.model.get_layer('user_bias').trainable = False
    student_model.model.get_layer('prob_bias').trainable = False


student_model.compile_model()

if 0:
    user_embedding = student_model.model.get_layer('user_embedding').get_weights()[0]
    prob_embedding = student_model.model.get_layer('prob_embedding').get_weights()[0]
    user_bias = student_model.model.get_layer('user_bias').get_weights()[0]
    prob_bias = student_model.model.get_layer('prob_bias').get_weights()[0]
    print('embeddings from NN after complining:')
    print('prob_bias', prob_bias[0:10])
    print('user_bias', user_bias[0:10])
    print('prob_embedding', prob_embedding[0:10,:])
    print('user_embedding', user_embedding[0:10,:])

history = student_model.fit(train_gen,
                  epochs=epochs,
                  val_gen=val_gen,
                  verbose=verbose,
                  filepath_bestmodel=best_model_file,
                  filepath_log=train_log)

student_model.load_weights(best_model_file)

result = student_model.evaluate(test_gen, metrics=['auc','acc','pre'], verbose=verbose, filepath_log=eval_log)
