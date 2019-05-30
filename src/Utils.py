import pandas as pd
import random
import numpy as np



def split_dataset_with_skill_prob_user_flag(data, validation_rate, testing_rate, shuffle=True):
    def split(dt):
        # user,  skill, prob
        return [[ [value[0], value[1], value[3], value[4]] for value in seq] for seq in dt], [[value[2] for value in seq] for seq in dt]

    seqs = data
    if shuffle:
        random.shuffle(seqs)

    # Get testing data
    test_idx = random.sample(range(0, len(seqs)-1), int(len(seqs) * testing_rate))
    X_test, y_test = split([value for idx, value in enumerate(seqs) if idx in test_idx])
    seqs = [value for idx, value in enumerate(seqs) if idx not in test_idx]

    # Get validation data
    val_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * validation_rate))
    X_val, y_val = split([value for idx, value in enumerate(seqs) if idx in val_idx])

    # Get training data
    X_train, y_train = split([value for idx, value in enumerate(seqs) if idx not in val_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_dataset(data, validation_rate, testing_rate, shuffle=True):
    def split(dt):
        return [[value[0] for value in seq] for seq in dt], [[value[1] for value in seq] for seq in dt]

    seqs = data
    if shuffle:
        random.shuffle(seqs)

    # Get testing data
    test_idx = random.sample(range(0, len(seqs)-1), int(len(seqs) * testing_rate))
    X_test, y_test = split([value for idx, value in enumerate(seqs) if idx in test_idx])
    seqs = [value for idx, value in enumerate(seqs) if idx not in test_idx]

    # Get validation data
    val_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * validation_rate))
    X_val, y_val = split([value for idx, value in enumerate(seqs) if idx in val_idx])

    # Get training data
    X_train, y_train = split([value for idx, value in enumerate(seqs) if idx not in val_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test

def read_file_prob(dataset_path):
    item_col_name = 'skill_id'
    data = pd.read_csv(dataset_path)
    print(dataset_path, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))

    students_seq = data.groupby("user_id", as_index=True)[item_col_name, "correct"].apply(lambda x: x.values.tolist()).tolist()

    # Step 3 - Rearrange the prob_id
    seqs_by_student = {}
    prob_ids = {}
    num_item = 0

    for seq_idx, seq in enumerate(students_seq):
        for (prob, answer) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            if prob not in prob_ids:
                prob_ids[prob] = num_item
                num_item += 1

            seqs_by_student[seq_idx].append((prob_ids[prob], answer))

    return list(seqs_by_student.values()), num_item


def generate_user_data(data):
    students_seq = data.groupby("user_id", as_index=True)['user_id', "skill_id", "correct", 'problem_id', 'test'].apply(lambda x: x.values.tolist()).tolist()
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0
    for seq_idx, seq in enumerate(students_seq):
        for (user, skill, answer, problem, test_flag) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            seqs_by_student[seq_idx].append((user, skill, answer, problem, test_flag))
    return list(seqs_by_student.values())

def read_file_with_skill_prob_user(dataset_path):
    data = pd.read_csv(dataset_path)
    print(dataset_path, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))
    students_seq = data.groupby("user_id", as_index=True)['user_id', "skill_id", "correct", 'problem_id'].apply(lambda x: x.values.tolist()).tolist()
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0
    for seq_idx, seq in enumerate(students_seq):
        for (user, skill, answer, problem) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            seqs_by_student[seq_idx].append((user, skill, answer, problem))
    return list(seqs_by_student.values()), {'num_entries': data.shape[0], 'num_users': len(np.unique(data['user_id'])), 'num_skills': len(np.unique(data['skill_id'])),
                             'num_probs': len(np.unique(data['problem_id']))}


def read_file_for_mf(dataset_path):
    data = pd.read_csv(dataset_path)
    print(dataset_path, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))
    return data[['user_id','problem_id', 'correct', 'skill_id']]


def read_file(dataset_path):
    data = pd.read_csv(dataset_path, dtype={'skill_name': str})

    # Step 1 - Remove problems without a skill_id
    data.dropna(subset=['skill_id'], inplace=True)

    data.correct = data.correct.astype(np.int)
    data.sort_values(by=['order_id'], inplace=True)

    print(dataset_name, data.shape[0],  len(np.unique(data['user_id'])), len(np.unique(data['problem_id'])), len(np.unique(data['skill_id'])))

    # Step 2 - Convert to sequence by student id
    students_seq = data.groupby("user_id", as_index=True)["skill_id", "correct"].apply(lambda x: x.values.tolist()).tolist()

    # Step 3 - Rearrange the skill_id
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0

    for seq_idx, seq in enumerate(students_seq):
        for (skill, answer) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            if skill not in skill_ids:
                skill_ids[skill] = num_skill
                num_skill += 1

            seqs_by_student[seq_idx].append((skill_ids[skill], answer))

    return list(seqs_by_student.values()), num_skill
