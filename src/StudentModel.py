from keras.callbacks import ModelCheckpoint, Callback, CSVLogger, Progbar
from keras.models import Sequential, Model
from keras.layers import TimeDistributed, Masking, Dense, Dropout, Input, Embedding, Add, Dot, Activation, Reshape, Multiply, Concatenate
from keras.layers.recurrent import LSTM
from keras import backend as K
from keras import regularizers, optimizers
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import random
import math
import numpy as np


# This method is for internal use. You should not use it outside of this file.
def model_evaluate(test_gen, model, metrics, verbose=0):
    def predict():
        def get_target_skills(preds, labels):
            target_skills = labels[:, :, 0:test_gen.num_skills]
            target_labels = labels[:, :, 0]

            target_preds = np.sum(preds * target_skills, axis=2)

            return preds, target_labels

        y_true_t = []
        y_pred_t = []
        test_gen.reset()

        while not test_gen.done:
            # Get batch
            batch_features, batch_users, batch_probs, batch_flags,  batch_labels = test_gen.next_batch()

            # Predict
            predictions = model.predict_on_batch([batch_features, batch_users, batch_probs, batch_flags])

            # Get target skills

            target_preds, target_labels = get_target_skills(predictions, batch_labels)
            target_test_flag = batch_flags[:, :, 0]

            flat_pred = np.reshape(target_preds, [-1])
            flat_true = np.reshape(target_labels, [-1])
            flat_test_flag = np.reshape(target_test_flag, [-1])


            # Remove mask
            mask_idx = np.where(flat_true == -1.0)[0]
            flat_pred = np.delete(flat_pred, mask_idx)
            flat_true = np.delete(flat_true, mask_idx)
            flat_test_flag = np.delete(flat_test_flag, mask_idx)
            # remove training entries
            train_mask_idx = np.where(flat_test_flag==0)[0]
            flat_pred = np.delete(flat_pred, train_mask_idx)
            flat_true = np.delete(flat_true, train_mask_idx)
            flat_test_flag = np.delete(flat_test_flag, train_mask_idx)
            # Save it
            y_true_t.extend(flat_true)
            y_pred_t.extend(flat_pred)


            if verbose and test_gen.step < test_gen.total_steps:
                progbar.update(test_gen.step)

        return y_true_t, y_pred_t

    assert (isinstance(test_gen, DataGenerator))
    assert (model is not None)
    assert (metrics is not None)

    if verbose:
        print("==== Evaluation Started ====")

    progbar = Progbar(target=test_gen.total_steps, verbose=verbose)

    y_true, y_pred = predict()


    bin_pred = [1 if p > 0.5 else 0 for p in y_pred]

    results = {}
    if 'auc' in metrics:
        results['auc'] = roc_auc_score(y_true, y_pred)
    if 'acc' in metrics:
        results['acc'] = accuracy_score(y_true, bin_pred)
    if 'pre' in metrics:
        results['pre'] = precision_score(y_true, bin_pred)

    if verbose:
        progbar.update(test_gen.step, results.items())
        print("==== Evaluation Done ====")

    return results

# This class is for internal use. You should not use it outside of this file.
class MetricsCallback(Callback):
    def __init__(self, data_gen, metrics, verbose=0):
        super(MetricsCallback, self).__init__()
        assert (isinstance(data_gen, DataGenerator))
        assert (metrics is not None)

        self.data_gen = data_gen
        self.metrics = metrics
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        if 'auc' in self.metrics:
            self.params['metrics'].append('val_auc')
        if 'acc' in self.metrics:
            self.params['metrics'].append('val_acc')
        if 'pre' in self.metrics:
            self.params['metrics'].append('val_pre')

    def on_epoch_end(self, epoch, logs={}):
        results = model_evaluate(self.data_gen, self.model, metrics=['auc', 'acc', 'pre'], verbose=self.verbose)

        if 'auc' in self.metrics:
            logs['val_auc'] = results['auc']
        if 'acc' in self.metrics:
            logs['val_acc'] = results['acc']
        if 'pre' in self.metrics:
            logs['val_pre'] = results['pre']

# This class defines the DKT model.
class DKTModel(object):
    def __init__(self, num_skills, num_features, num_probs, num_users, embedding_size, regularization = 1e-4, optimizer='rmsprop', hidden_units=100, batch_size=5, dropout_rate=0.5):
        def get_target_skills(y_true, y_pred):
            target_skills = y_true[:, :, 0:num_skills]
            target_labels = y_true[:, :, num_skills]
            target_preds = K.sum(y_pred * target_skills, axis=2)

            return target_preds, target_labels

        def loss_function(y_true, y_pred):
            target_preds, target_labels = get_target_skills(y_true, y_pred)
            return K.binary_crossentropy(target_labels, target_preds)

        self.batch_size = batch_size
        self.num_skills = num_skills
        self.num_users = num_users
        self.num_probs = num_probs
        self.embedding_size = embedding_size
        self.regularization = regularization
        self.optimizer = optimizer
        # this is DKT

        if 0:
            self.model = Sequential()
            self.model.add(Masking(-1., batch_input_shape=(batch_size, None, num_features)))
            self.model.add(LSTM(hidden_units, return_sequences=True, stateful=True))
            self.model.add(Dropout(dropout_rate))
            self.model.add(TimeDistributed(Dense(num_skills, activation='sigmoid')))
            self.model.compile(loss=loss_function, optimizer=optimizer)

        if 1:


        #user = Input(name = 'user', shape = [1])
        skill_correct_input = Input(name = 'skill_correct', shape = [None, num_features], batch_shape=[batch_size, None, num_features])
        prob_input = Input(name = 'prob', shape = [None, 1], batch_shape=[batch_size, None, 1])
        user_input = Input(name = 'user', shape = [None, 1], batch_shape=[batch_size, None, 1])
        flag_input = Input(name = 'flag', shape = [None, 1], batch_shape=[batch_size, None, 1])


        skill_correct = Masking(mask_value = -1, batch_input_shape=(batch_size, None, num_features)) (skill_correct_input)
        prob = Masking(name='prob_masking', mask_value = -1, batch_input_shape=(batch_size, None, 1)) (prob_input)
        user = Masking(name='user_masking', mask_value = -1, batch_input_shape=(batch_size, None, 1)) (user_input)
        # flag = Masking(name='user_masking', mask_value = -1, batch_input_shape=(batch_size, None, 1)) (flag_input)
        if 1:
            user_embedding = Embedding(name = 'user_embedding',
                                       input_dim = self.num_users,
                                       output_dim = self.embedding_size,
                                       embeddings_regularizer=regularizers.l2(self.regularization))(user)

            user_bias = Embedding(name = 'user_bias',
                                       input_dim = self.num_users,
                                       output_dim = 1,
                                       embeddings_regularizer=regularizers.l2(self.regularization))(user)

            prob_embedding = Embedding(name = 'prob_embedding',
                                       input_dim = self.num_probs,
                                       output_dim = self.embedding_size,
                                       embeddings_regularizer=regularizers.l2(self.regularization))(prob)

            prob_bias = Embedding(name = 'prob_bias',
                                       input_dim = self.num_probs,
                                       output_dim = 1,
                                       embeddings_regularizer=regularizers.l2(self.regularization))(prob)
        if 0:
            user_embedding = Dense(name='user_embedding', units=self.embedding_size, use_bias=False) (user)
            prob_embedding = Dense(name='prob_embedding', units=self.embedding_size, use_bias=False) (prob)

            user_bias = Dense(name='user_bias', units=1, use_bias=False) (user)
            prob_bias = Dense(name='prob_bias', units=1, use_bias=False) (prob)

        user_embedding_reshape = Reshape(name='user_embedding_reshape', target_shape=(-1, self.embedding_size)) (user_embedding)
        prob_embedding_reshape = Reshape(name='prob_embedding_reshape', target_shape=(-1, self.embedding_size)) (prob_embedding)
        user_bias_reshape = Reshape(name='user_bias_reshape', target_shape=(-1, 1)) (user_bias)
        prob_bias_reshape = Reshape(name='prob_bias_reshape', target_shape=(-1, 1)) (prob_bias)

        LSTM_input = Concatenate(axis=2)([prob_embedding_reshape, skill_correct, prob_bias_reshape])

        skill_correct_LSTM = LSTM(name='LSTM', units=hidden_units, return_sequences=True, stateful=True, recurrent_dropout=0.5)(LSTM_input)

        dropout = Dropout(name='dropout', rate= dropout_rate)(skill_correct_LSTM)
        user_dyn = Dense(name='user_dyn_from_LSTM', units=self.embedding_size, activation='relu')(dropout)

        user_merge = Add(name='add_user_dyn_and_stat')([user_dyn, user_embedding_reshape])

        # inner_prod = Dot(name='dot_product', normalize=False, axes=2)([user_merge, prob_embedding_reshape])

        # This is actually inner product layer
        multiply_user_prob = Multiply(name='multiply_user_prob')([user_merge, prob_embedding_reshape])
        inner_prod = Dense(name='inner_prod', units=1, activation='linear', kernel_initializer="one", use_bias=False)(multiply_user_prob)
        inner_prod.trainable = False

        merged = Add(name='add_dot_and_bias')([inner_prod, user_bias_reshape, prob_bias_reshape])

        out = Activation(name='sigmoid_output',activation='sigmoid')(merged)
        # out = Reshape(name='output_reshape', target_shape=(-1, 1)) (out)

        self.model = Model(inputs=[skill_correct_input, user_input, prob_input, flag_input], outputs=out)
        self.compile_model()


    def compile_model(self):
        self.model.compile(loss = 'binary_crossentropy', metrics=['accuracy'], optimizer=self.optimizer)

    def load_weights(self, filepath):
        assert(filepath is not None)
        self.model.load_weights(filepath)

    def fit(self, train_gen, epochs, val_gen, verbose=0, filepath_bestmodel=None, filepath_log=None):
        assert (isinstance(train_gen, DataGenerator))
        assert (isinstance(val_gen, DataGenerator))

        callbacks = []
        callbacks.append(MetricsCallback(val_gen, metrics=['auc','pre','acc']))

        if filepath_bestmodel is not None:
            callbacks.append(ModelCheckpoint(filepath_bestmodel, monitor='val_loss', verbose=verbose, save_best_only=True))
        if filepath_log is not None:
            callbacks.append(CSVLogger(filepath_log))

        if verbose:
            print("==== Training Started ====")

        history = self.model.fit_generator(shuffle=False,
                                             validation_data=val_gen.get_generator(),
                                             validation_steps=val_gen.total_steps,
                                             epochs=epochs,
                                             steps_per_epoch=train_gen.total_steps,
                                             generator=train_gen.get_generator(),
                                             callbacks=callbacks,
                                             verbose=verbose)

        if verbose:
            print("==== Training Done ====")

        return history

    def evaluate(self, test_gen, metrics, verbose=0, filepath_log=None):
        assert (isinstance(test_gen, DataGenerator))
        assert (metrics is not None)

        results = model_evaluate(test_gen, self.model, metrics, verbose)

        if filepath_log is not None:
            with open(filepath_log, 'w') as fl:
                fl.write("auc,acc,pre\n{0},{1},{2}".format(results['auc'], results['acc'], results['pre']))

        return results


# This class is responsible for feeding the data into the model following a specific format.
class DataGenerator(object):
    def __init__(self, features, labels, num_skills, num_users, num_probs, batch_size):
        self.features = features
        self.labels = labels
        self.num_skills = num_skills
        self.num_probs = num_probs
        self.num_users = num_users
        self.batch_size = batch_size

        self.step = 0
        self.done = False
        self.feature_dim = num_skills * 2
        self.label_dim = 1
        self.user_dim = 1
        self.prob_dim = 1
        self.features_len = len(features)
        self.total_steps = int(math.ceil(float(self.features_len) / self.batch_size))
        self.feature_encoder = OneHotEncoder(self.feature_dim, sparse=False)
        self.label_encoder = OneHotEncoder(self.label_dim, sparse=False)
        self.user_encoder = OneHotEncoder(self.user_dim, sparse=False)
        self.prob_encoder = OneHotEncoder(self.prob_dim, sparse=False)
        #self.feature_encoder = OneHotEncoder(categories=[range(self.feature_dim)], sparse=False)
        #self.label_encoder = OneHotEncoder(categories=[range(self.label_dim)], sparse=False)

    # Ref: https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
    def __pad_sequences(self, sequences, maxlen=None, dim=1, dtype='int32', padding='pre', truncating='pre', value=0.):
        '''
            Override keras method to allow multiple feature dimensions.

            @dim: input feature dimension (number of features per timestep)
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x

    def next_batch(self):
        def fill_batches(x, user, prob, flag, y):
            for e in range(self.batch_size - len(x)):
                x.append([np.array([-1.0 for _ in range(0, self.feature_dim)])])
                y.append([np.array([-1.0 for _ in range(0, self.label_dim)])])
                user.append([np.array([-1.0 for _ in range(0, self.user_dim)])])
                prob.append([np.array([-1.0 for _ in range(0, self.prob_dim)])])
                flag.append([np.array([-1.0 for _ in range(0, 1)])])
            return x, user, prob, flag, y

        def pad_sequences(x, user, prob, flag, y):
            max_seq_steps = max([len(seq) for seq in x])
            x = self.__pad_sequences(x, padding='post', maxlen=max_seq_steps, dim=self.feature_dim, value=-1.0, dtype='float')
            y = self.__pad_sequences(y, padding='post', maxlen=max_seq_steps, dim=self.label_dim, value=-1.0, dtype='float')
            user = self.__pad_sequences(user, padding='post', maxlen=max_seq_steps, dim=self.user_dim, value=-1.0, dtype='float')
            prob = self.__pad_sequences(prob, padding='post', maxlen=max_seq_steps, dim=self.prob_dim, value=-1.0, dtype='float')
            flag = self.__pad_sequences(flag, padding='post', maxlen=max_seq_steps, dim=1, value=-1.0, dtype='float')
            return x, user, prob, flag, y

        def encode_batch(batch_questions, batch_answers):
            x = [] # skill_correct
            y = [] # correct
            prob = []
            user = []
            test_flag = []

            for idx, questions in enumerate(batch_questions):
                x_student = []
                y_student = []
                prob_student = []
                user_student = []
                test_flag_student = []

                x_data = np.zeros(self.feature_dim, dtype=int)
                answers = batch_answers[idx]

                # should move one step ahead to generate the response
                if len(questions) > 1:
                    for skill_index in range(len(questions)-1):
                        answer = answers[skill_index]
                        skill_value = questions[skill_index][1]
                        # Encode skill_id

                        skill_answer = skill_value * 2 + answer
                        # print('skill_answer')
                        # print(skill_answer)
                        skill_answer = np.array([skill_answer])
                        skill_value = np.array([skill_value])
                        skill_answer = skill_answer.reshape(-1,1)
                        skill_value = skill_value.reshape(-1,1)
                        x_data = self.feature_encoder.fit_transform(skill_answer)[0]
                        x_student.append(x_data)
                        answer2 = answers[skill_index+1]
                        # skill_value2 = questions[skill_index+1]
                        # skill_value2 = np.array([skill_value2])
                        # skill_value2 = skill_value2.reshape(-1,1)

                        # y_data = self.label_encoder.fit_transform(skill_value2)[0]
                        y_data = [answer2]
                        y_student.append(y_data)

                        if 0: # memory issue
                            user_id = np.array([questions[skill_index+1][0]])
                            user_id = user_id.reshape(-1, 1)
                            user_data = self.user_encoder.fit_transform(user_id)[0]
                            user_student.append(user_data)

                            prob_id = np.array([questions[skill_index+1][2]])
                            prob_id = prob_id.reshape(-1, 1)
                            prob_data = self.prob_encoder.fit_transform(prob_id)[0]
                            prob_student.append(prob_data)

                        user_data = [questions[skill_index+1][0]]
                        user_student.append(user_data)
                        prob_data = [questions[skill_index+1][2]]
                        prob_student.append(prob_data)

                        test_flag_data = [questions[skill_index+1][3]]
                        test_flag_student.append(test_flag_data)


                    x.append(x_student)
                    y.append(y_student)
                    user.append(user_student)
                    prob.append(prob_student)
                    test_flag.append(test_flag_student)

            return x, user, prob, test_flag, y

        assert(~self.done)

        start_pos = self.step * self.batch_size
        end_pos = (self.step + 1) * self.batch_size

        if end_pos >= self.features_len:
            self.done = True
            end_pos = self.features_len

        # Apply one-hot encoding
        x_batch, user_batch, prob_batch, flag_batch, y_batch = encode_batch(self.features[start_pos:end_pos], self.labels[start_pos:end_pos])

        # Fill up incomplete batch
        x_batch, user_batch, prob_batch, flag_batch, y_batch = fill_batches(x_batch, user_batch, prob_batch, flag_batch, y_batch)

        # Pad sequences to the same size
        x_batch, user_batch, prob_batch, flag_batch, y_batch = pad_sequences(x_batch, user_batch, prob_batch, flag_batch, y_batch)

        self.step += 1

        return x_batch, user_batch, prob_batch, flag_batch, y_batch

    def reset(self, shuffle=True):
        if shuffle:
            self.shuffle()

        self.done = False
        self.step = 0

    def shuffle(self):
        combined = list(zip(self.features, self.labels))
        random.shuffle(combined)
        self.features[:], self.labels[:] = zip(*combined)

    def get_generator(self):
        while True:
            self.reset()
            while not self.done:
                batch_features, batch_users, batch_probs, batch_flags, batch_labels = self.next_batch()
                # (batch_labels.shape)
                yield [batch_features, batch_users, batch_probs, batch_flags], batch_labels
