#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 18:31:56 2019

@author: lxu
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
    
# dataset = "~/Documents/StudentLearningProcess/skill_builder_data_corrected_withskills.csv" # Dataset path
# dataset = "~/Documents/StudentLearningProcess/2015_100_skill_builders_main_problems.csv" # Dataset path
dataset = '~/Documents/Assistment09-problem-single_skill.csv'
# dataset = "data/skill_builder_data.csv"
best_model_file = "saved_models/ASSISTments.best.model.weights.hdf5" # File to save the model.
# best_model_file = "logs/8/saved_models/ASSISTments.best.model.weights.hdf5" # File to save the model.
train_log = "logs/dktmodel.train.log" # File to save the training log.
eval_log = "logs/dktmodel.eval.log" # File to save the testing log.
optimizer = "adagrad" # Optimizer to use
lstm_units = 250 # Number of LSTM units
batch_size = 40 # Batch size
epochs = 10 # Number of epochs to train
dropout_rate = 0.6 # Dropout rate
verbose = 1 # Verbose = {0,1,2}
testing_rate = 0.2 # Portion of data to be used for testing
validation_rate = 0.2 # Portion of training data to be used for validation

from Utils import *

dataset, stas = read_file_with_skill_prob_user(dataset)
num_skills = stas['num_skills']
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset_with_skill_prob_user(dataset, validation_rate, testing_rate)

print("======== Data Summary ========")
print("Data size: %d" % len(dataset))
print("Training data size: %d" % len(X_train))
print("Validation data size: %d" % len(X_val))
print("Testing data size: %d" % len(X_test))
print("Number of skills: %d" % num_skills)
print("==============================")

from StudentModel import DKTModel, DataGenerator

# Create generators for training/testing/validation
train_gen = DataGenerator(X_train, y_train, num_skills, batch_size)
val_gen = DataGenerator(X_val, y_val, num_skills, batch_size)
test_gen = DataGenerator(X_test, y_test, num_skills, batch_size)

x, User, Prob,  y = train_gen.next_batch()

print('x, user, prob, y:', x, User, Prob, y)
# Create model
student_model = DKTModel(num_skills=train_gen.num_skills,
                      num_features=train_gen.feature_dim,
                      optimizer=optimizer,
                      hidden_units=lstm_units,
                      batch_size=batch_size,
                      dropout_rate=dropout_rate)


history = student_model.fit(train_gen,
                  epochs=epochs,
                  val_gen=val_gen,
                  verbose=verbose,
                  filepath_bestmodel=best_model_file,
                  filepath_log=train_log)

student_model.load_weights(best_model_file)

result = student_model.evaluate(test_gen, metrics=['auc','acc','pre'], verbose=verbose, filepath_log=eval_log)
