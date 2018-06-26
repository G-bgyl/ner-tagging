import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


import pickle
import time
from datetime import datetime
from models.data_utils import Clean_data
NUM = "$NUM$"
class Model():

    def __init__(self,config):
        """

        :param config: class Config instance from config.py

        """
        self.config = config
        self.clean = Clean_data()

        '''
        # get from pretrained embeddings
        embeddings = {}

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)

        '''
    def generate_new(self,unk,vocab_str, vocab_vec):
        """Dealing with Unknown words in zh_w2v, append new word string into vocab_str, new word vector into vocab_vec
          Input: unknown word,global vocab_str,vocab_vec
          Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
        """
        s = np.vstack(vocab_vec)
        v = np.var(s, 0)  # distributional parameter for zh_w2v, for later generating random embedding for UNK
        m = np.mean(s, 0)
        RS = np.random.RandomState()

        unk_vec =RS.multivariate_normal(m, np.diag(v))
        vocab_str.append(unk)
        vocab_vec.append(unk_vec)

        return [unk_vec], vocab_str, vocab_vec

    def sentence2sequence(self,sentence, vocab_str, vocab_vec):
        """
            - Turns an input paragraph into an (m,d) matrix,
                where n is the number of sentence length
                and d is the number of dimensions each word vector has.
              Input: sentence, string
              Output: list of index of each word segment in sentence
        """
        sent_raw_list = sentence
        sent_str_list = []
        sent_idx_list = []

        for j in range(len(sent_raw_list)):
            if len(sent_idx_list) < self.config.maxSeqLength:
                word = sent_raw_list[j]
                try:
                    id = vocab_str.index(word)
                    sent_idx_list.append(vocab_vec[id])
                    sent_str_list.append(word)
                except ValueError:
                    print('generate new')
                    unk_vec, vocab_str, vocab_vec = self.generate_new(word, vocab_str, vocab_vec)
                    sent_str_list.extend(word)
                    sent_idx_list.extend(unk_vec)
            else:
                break
        # make sure the length of two return params are fixed.
        # Padding
        left_len = self.config.maxSeqLength - len(sent_idx_list)
        if left_len > 0:
            sent_idx_list.extend(np.zeros((self.config.numDimensions,)) for i in range(left_len))
            sent_str_list.extend([''] * left_len)
        print('finish one sent')
        return sent_str_list, sent_idx_list


    def contextualize(self,raw_data, vocab_str=None, vocab_vec=None, save = 'final_data', overide=None):
        ''' read in raw_data and prepare final data
        :param raw_data: output of read_data, pandas data_frame
        :param overide: default None, it not none, input a pickle file name
        :return: final data contains (labels,sent_str_list,sent_idx_list)
        '''
        contextualize_start_time = time.time()
        if overide:
            data = pickle.load(open("datas/%s"%(overide), "rb"))
            print('finish load final_data from pickle!')
            return data
        else:
            print('Begin prepare for final data...')
            final_data = []
            i = 0
            for index, each in raw_data.iterrows():
                print('iter throw raw data')
                id, list_content, labels, one_hot = each
                list_content = self.clean.sent_num_mask(one_sent=list_content)
                one_hot = eval(one_hot)

                sent_str_list, sent_idx_list = self.sentence2sequence(list_content, vocab_str, vocab_vec)
                final_data.append((one_hot, sent_str_list, sent_idx_list))
                if i % 50 == 0:
                    print(i, len(sent_str_list), len(sent_idx_list[0]), 'process sms message in sentence2sequence!')
                i += 1
            pickle.dump(final_data, open("datas/%s-%s.p"%(save, datetime.now().strftime("%m%d")), "wb+"))

            contextualize_finish_time = time.time()
            spend_time = float((contextualize_finish_time - contextualize_start_time) / 60)
            print("======== total spend for contextualize final data:", spend_time, 'minutes ========\n')

            return np.asarray(final_data)

    def split_final(self,final_data, train_size=0, overide=None):
        '''split data set into train, cross validation, test.

        :param final_data: (id,raw_content, scene_category), None when overide is True
        :param train_size: if = 0 , means use full length of training. if = 0.9, means only use 10% of training set.
        :param overide: False or None if not overide; a time with month and date(eg.0801) if overide to get that date's data.
        :return: seperate data into train, dev, test
        '''

        if overide:
            train_data = pickle.load(open("datas/train_data%s.p"%(overide), "rb"))
            cv_data = pickle.load(open("datas/cv_data%s.p"%(overide), "rb"))
            test_data = pickle.load(open("datas/test_data%s.p"%(overide), "rb"))

            print('finish load split data from pickle!')
            return train_data, cv_data, test_data
        else:
            train_dev, test_data = train_test_split(final_data, test_size=0.2, random_state=42)
            train_data, cv_data = train_test_split(train_dev, test_size=0.25, random_state=42)
            train_data, _ = train_test_split(train_data, test_size=train_size, random_state=42)

            today = datetime.now().strftime("%m%d")

            pickle.dump(train_data, open("datas/train_data%s.p"%(today), "wb+"))
            pickle.dump(cv_data, open("datas/cv_data%s.p"%(today), "wb+"))
            pickle.dump(test_data, open("datas/test_data%s.p"%(today), "wb+"))

            print('finish split data!')
            return train_data, cv_data, test_data

    def turn_list_into_nparray(self,list):
        """return an array instead of a list"""
        np_list = np.asarray(list)

        return np_list

    def convert_batch_into_array(self,batch):
        """convert list of batch into array"""
        new_batch = []
        for each in batch:
            new_batch.append(self.turn_list_into_nparray(each))
        return np.asarray(new_batch)

    def add_padding(self, batch, shape):
        """ padding batch to shape

        :param batch: with shape of batchsize, shape[0], shape[1]
        :param shape: maxSeqLength, numClasses
        :return:
        """
        new_batch = []
        for each in batch:
            pad = shape[0]-len(each)
            # for i in range(shape[0]-len(each)):
            each = np.vstack((each, np.zeros((pad, shape[1]))))
            new_batch.append(each)
            if len(each) > shape[0]:
                raise KeyError('sms longer than %s'%(shape[0]))
        # print([(i,x) for i,x in enumerate(new_batch) if x.shape !=tuple(shape)])

        return np.asarray(new_batch)
    def sum_label(self):
        '''
        # calculate f1 score
        tp, fp, fn, tn = 0, 0, 0, 0

        # for scene_code in target_scene:
        # type_to_eval=1 when test S0002
        type_to_eval = target_scene[scene_code]
        for i in range(len(correctPred_list)):
            pred = list(prediction_list[i])
            accu = list(nextBatchLabels[i])
            if pred.index(max(pred)) == type_to_eval:

                if accu.index(1) == type_to_eval:
                    tp += 1
                else:
                    fn += 1
            else:
                if accu.index(1) == type_to_eval:
                    fp += 1
                else:
                    tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy_ = (tp + tn) / (tp + fp + tn + fn)
        f1_score = 2 / (1 / precision + 1 / recall)
        print('............ stats for %s ............' % (scene_code))
        print('tp', tp)
        print('fp', fp)
        print('fn', fn)
        print('tn', tn)
        print('Precision for %s:' % (scene_code), precision)
        print('Recall for %s:' % (scene_code), recall)
        print('Accuracy for %s:' % (scene_code), accuracy_)
        print('F1 for %s:' % (scene_code), f1_score)
        return precision, recall, accuracy_, f1_score
        '''
        pass
    def summary(self):
        pass

    def train(self, train_data, save = None):
        tf.reset_default_graph()
        with tf.variable_scope("lstm"):
            # None, self.config.maxSeqLength, self.config.numClasses
            labels = tf.placeholder(tf.int32, [self.config.batchSize, self.config.maxSeqLength, self.config.numClasses],name='labels')

            batch_vec_data = tf.placeholder(
                tf.float32, [self.config.batchSize, self.config.maxSeqLength, self.config.numDimensions], "batch_vec_data")  # batchSize

            lstmCell = tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)
            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.config.drop_out)


            value, _ = tf.nn.dynamic_rnn(lstmCell, batch_vec_data, dtype=tf.float32)

        with tf.variable_scope("proj"):
            weight = tf.Variable(tf.truncated_normal([self.config.lstmUnits, self.config.numClasses]),name='weight')
            bias = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]),name = 'bias')

            value = tf.reshape(value, [-1, self.config.lstmUnits])
            predict = tf.matmul(value, weight) + bias
            prediction = tf.reshape(predict,[self.config.batchSize, self.config.maxSeqLength, self.config.numClasses])

        with tf.variable_scope("pred"):
            if self.config.use_crf:
                print(prediction.shape)
                print(labels.shape)
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    prediction, labels, self.config.batchSize)
                # self.trans_params = trans_params  # need to evaluate it for decoding
                loss = tf.reduce_mean(-log_likelihood)
            else:
                correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
                    # + self.config.beta * tf.nn.l2_loss(weight) + self.config.beta * tf.nn.l2_loss(bias)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.9, beta2=0.999).minimize(loss)

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # s_w = tf.summary.FileWriter('tensorboard',sess.graph)
        convertion = 0


        # begin training
        for i in range(self.config.iterations):
            # detect convergence
            if convertion > 90:
                print('\nIteration %s Successfully converge!' % (i))
                break

            batch = np.random.randint(len(train_data), size=self.config.batchSize)
            batch_data = [train_data[k] for k in batch]

            # zip(*batch_data) returns labels, sent_str_list, sent_idx_list
            nextBatchLabels, _, nextBatchVec = zip(*batch_data)
            nextBatchLabels = self.convert_batch_into_array(nextBatchLabels)
            nextBatchLabels = self.add_padding(nextBatchLabels, [self.config.maxSeqLength, self.config.numClasses])
            feed_dict = {batch_vec_data: nextBatchVec, labels: nextBatchLabels}
            # print([(i,x.shape)  for i, x in enumerate(nextBatchLabels) if x.shape !=(150,10)])
            # print([(i,len(x))  for i, x in enumerate(nextBatchVec) if len(x) != 150])

            _, accuracy_num, correctPred_list,loss_ = sess.run([optimizer,accuracy, correctPred,loss], feed_dict=feed_dict)
            # calculate accuracy
            train_accu = (accuracy_num * 100)
            if (i + 1) % 50 == 0:
                self.summary()
                print("Batch %s\t: acc \t %s %% ,loss\t %s " % ((i + 1, train_accu, loss_)))

            # help made judgment of converging.
            if 100 - train_accu < 0.25:
                convertion += 1

        if save:
            today = datetime.now().strftime("%m%d")
            # Save the variables to disk.
            save_path = saver.save(sess, "tf_model/model-%s%s.ckpt" % (save,today))
            print("Model saved in path: %s" % save_path)
            #
            # print("saved to models/model-%s%s.p" % (save,today))

    '''

    def get_pretrained_embeddings(self):
        # -----------------------------------
        # get the word embedding of each word
        # -----------------------------------

        # shape = (batch, sentence, word_vector_size)
        pretrained_embeddings = tf.nn.embedding_lookup(self.L, self.word_ids)

        return pretrained_embeddings


    def word_representation(self):
        # -----------------------------------
        # create a word representation using bidirectional LSTM model
        # -----------------------------------

        # shape = (batch size, max length of sentence, max length of word)
        char_ids = tf.placeholder(tf.int32, shape=[None, None, None])

        # shape = (batch_size, max_length of sentence)
        word_lengths = tf.placeholder(tf.int32, shape=[None, None])


        # 1. get character embeddings
        K = tf.get_variable(name="char_embeddings", dtype=tf.float32,
            shape=[self.nchars, self.dim_char])
        # shape = (batch, sentence, word, dim of char embeddings)
        char_embeddings = tf.nn.embedding_lookup(K, char_ids)

        # 2. put the time dimension on axis=1 for dynamic_rnn
        s = tf.shape(char_embeddings) # store old shape
        # shape = (batch x sentence, word, dim of char embeddings)
        char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], s[-1]])
        word_lengths = tf.reshape(self.word_lengths, shape=[-1])

        # 3. bi lstm on chars
        cell_fw = tf.contrib.rnn.LSTMCell(self.char_hidden_size, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(self.char_hidden_size, state_is_tuple=True)

        _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
            cell_bw, char_embeddings, sequence_length=word_lengths,
            dtype=tf.float32)
        # shape = (batch x sentence, 2 x char_hidden_size)
        output = tf.concat([output_fw, output_bw], axis=-1)

        # shape = (batch, sentence, 2 x char_hidden_size)
        char_rep = tf.reshape(output, shape=[-1, s[1], 2*self.char_hidden_size])
        return char_rep


    def get_full_rep(self):

        pretrained_embeddings = self.get_pretrained_embeddings()
        char_rep  = self.word_representation()


        # shape = (batch, sentence, 2 x char_hidden_size + word_vector_size)
        word_embeddings = tf.concat([pretrained_embeddings, char_rep], axis=-1)
        return word_embeddings


    def contextual_word_representation(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
        word_embeddings = self.get_full_rep()
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw, word_embeddings,
                                                                    sequence_length=self.sequence_lengths,
                                                                    dtype=tf.float32)

        context_rep = tf.concat([output_fw, output_bw], axis=-1)
        return context_rep

    def decoding(self):
        context_rep = self.contextual_word_representation()
        W = tf.get_variable("W", shape=[2 * self.hidden_size, self.num_class],
                            dtype=tf.float32)

        b = tf.get_variable("b", shape=[self.num_class], dtype=tf.float32,
                            initializer=tf.zeros_initializer())

        ntime_steps = tf.shape(context_rep)[1]
        context_rep_flat = tf.reshape(context_rep, [-1, 2 * self.hidden_size])
        pred = tf.matmul(context_rep_flat, W) + b
        scores = tf.reshape(pred, [-1, ntime_steps, self.num_class])
        
        '''
