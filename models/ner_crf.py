import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


import pickle
import time
from datetime import datetime


class Model():

    def __init__(self,config):
        """

        :param config: class Config instance from config.py

        """
        self.config = config

        '''
        # get from pretrained embeddings
        embeddings = {}

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)

        '''
    def generate_new(self,unk, vocab_str, vocab_vec):
        """Dealing with Unknown words in zh_w2v, append new word string into vocab_str, new word vector into vocab_vec
          Input: unknown word,global vocab_str,vocab_vec
          Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
        """
        word = unk
        # fake new vector
        well = False
        print(vocab_vec[-1],len(vocab_vec[-1]))
        s = np.vstack(vocab_vec)
        Zvar = np.var(s, 0)  # distributional parameter for zh_w2v, for later generating random embedding for UNK
        Zmean = np.mean(s, 0)
        RS = np.random.RandomState()


        # Greedy search for word: if it did not find the whole word in zh_w2v, then truncate tail to find
        i = len(unk)
        tmp_vec_for_avg = []
        while len(unk) > 0:

            wd = unk[:i]  # shallow copy
            if wd in vocab_str:
                tmp_vec_for_avg.append(vocab_vec[vocab_str.index(wd)])
                unk = unk[i:]  # set to whatever the rest of words is, eg. hallway --> hall, way, split to 2 words
                i = len(unk)
            else:
                i = i - 1

            # finish greedy search. code below only run once for each time this function called, and
            # tmp_vec_for_avg should contain at least two vectors in order to sum without broken.

            if i == 0:
                # alternative: new_word_vec = np.average(tmp_vec_for_avg, axis=0)
                # maybe not rubust because of there are less than 2 list in tmp_vec_for_avg, means the greedy search
                # didn't used
                # new_word_vec = [sum(l) for l in zip(*tmp_vec_for_avg)]
                if tmp_vec_for_avg:
                    new_word_vec = np.average(tmp_vec_for_avg, axis=0)
                    vocab_str.append(word)
                    vocab_vec.append(new_word_vec)
                    well = True
                    break
                else:
                    new_word_vec = RS.multivariate_normal(Zmean, np.diag(Zvar))
                    vocab_str.append(word)
                    vocab_vec.append(new_word_vec)
                    break
        if not well:
            print('numbers')
        # maybe not rubust because the definition of new_word_vec is hidden in a if condition clause
        return vocab_str, vocab_vec, new_word_vec




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
                    sent_idx_list.append(vocab_vec[vocab_str.index(word)])
                    sent_str_list.append(word)
                except ValueError:
                    vocab_str, vocab_vec, new_word_vec = self.generate_new(word, vocab_str, vocab_vec)
                    sent_str_list.append(word)
                    sent_idx_list.append(new_word_vec)

            else:
                break
        # make sure the length of two return params are fixed.
        # Padding
        left_len = self.config.maxSeqLength - len(sent_idx_list)
        if left_len > 0:
            sent_idx_list.extend(np.zeros((300,)) for i in range(left_len))
            sent_str_list.extend([''] * left_len)
        return sent_str_list, sent_idx_list


    def contextualize(self,raw_data, vocab_str=None, vocab_vec=None, save = 'final_data', overide=None):
        ''' read in raw_data and prepare final data
        :param raw_data: output of read_data, pandas data_frame
        :param overide: default None, it not none, input a pickle file name
        :return: final data contains (labels,sent_str_list,sent_idx_list)
        '''
        contextualize_start_time = time.time()
        if overide:
            data = pickle.load(open(".datas/%s"%(overide), "rb"))
            print('finish load final_data from pickle!')
            return data
        else:
            final_data = []
            i = 0
            for index, each in raw_data.iterrows():

                id, list_content, labels = each
                list_content = eval(list_content)
                labels = eval(labels)

                sent_str_list, sent_idx_list = self.sentence2sequence(list_content, vocab_str, vocab_vec)
                final_data.append((labels, sent_str_list, sent_idx_list))
                if i % 500 == 0:
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
            train_data = pickle.load(open("DATA/train_data.p"%(overide), "rb"))
            cv_data = pickle.load(open("DATA/cv_data%s.p"%(overide), "rb"))
            test_data = pickle.load(open("DATA/test_data%s.p"%(overide), "rb"))

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


    def train(self,train_data,save = None):

        tf.reset_default_graph()
        labels = tf.placeholder(tf.float32, [None, self.config.numClasses])  # batchSize
        # maybe is a batch size data
        batch_vec_data = tf.placeholder(
            tf.float32, [None, self.config.maxSeqLength, self.config.numDimensions], "context")  # batchSize

        lstmCell = tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=self.config.drop_out)

        value, _ = tf.nn.dynamic_rnn(lstmCell, batch_vec_data, dtype=tf.float32)
        weight = tf.Variable(tf.truncated_normal([self.config.lstmUnits, self.config.numClasses]))
        bias = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]))
        value = tf.transpose(value, [1, 0, 2])
        last = tf.gather(value, int(value.get_shape()[0]) - 1)
        prediction = tf.matmul(last, weight) + bias

        correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels) + self.config.beta * tf.nn.l2_loss(
                weight) + self.config.beta * tf.nn.l2_loss(bias))
        optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999).minimize(loss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        convertion = 0


        # begin training
        for i in range(self.config.iterations):
            # detect convergence
            if convertion > 90:
                print('\nIteration %s Successfully converge!' % (i))
                break

            batch = np.random.randint(len(train_data), size=self.config.ibatchSize)
            batch_data = [train_data[k] for k in batch]

            # zip(*batch_data) returns labels, sent_str_list, sent_idx_list
            nextBatchLabels, _, nextBatchVec = zip(*batch_data)
            feed_dict = {batch_vec_data: nextBatchVec, labels: nextBatchLabels}
            _, accuracy_num, correctPred_list = sess.run([optimizer,accuracy, correctPred],
                                                      feed_dict=feed_dict)
            #calculate accuracy
            train_accu = (accuracy_num * 100)
            if (i + 1) % 50 == 0:
                print("Accuracy for batch %s : %s %% " % ((i + 1), train_accu))

            # help made judgment of converging.
            if 100 - train_accu < 0.25:
                convertion += 1

        if save:
            today = datetime.now().strftime("%m%d")
            pickle.dump(sess, open("models/model-%s%s.p" % (save,today), "wb+"))
            print("saved to models/model-%s%s.p" % (save,today))




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
