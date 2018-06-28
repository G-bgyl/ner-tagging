import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

import pickle
import time
from datetime import datetime
import csv

from models.data_utils import Clean_data


class Model():

    def __init__(self, config):
        """

        :param config: class Config instance from config.py

        """
        self.config = config
        self.clean = Clean_data()
        self.sess = None
        self.saver = None
        self.optimizer = None
        self.loss = None

        '''
        # get from pretrained embeddings
        embeddings = {}

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)

        '''

    def generate_new(self, unk, vocab_str, vocab_vec):
        """Dealing with Unknown words in zh_w2v, append new word string into vocab_str, new word vector into vocab_vec
          Input: unknown word,global vocab_str,vocab_vec
          Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
        """
        s = np.vstack(vocab_vec)
        v = np.var(s, 0)  # distributional parameter for zh_w2v, for later generating random embedding for UNK
        m = np.mean(s, 0)
        RS = np.random.RandomState()

        unk_vec = RS.multivariate_normal(m, np.diag(v))
        vocab_str.append(unk)
        vocab_vec.append(unk_vec)

        return [unk_vec], vocab_str, vocab_vec

    def sentence2sequence(self, sentence, vocab_str, vocab_vec):
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
                    unk_vec, vocab_str, vocab_vec = self.generate_new(word, vocab_str, vocab_vec)
                    sent_str_list.extend(word)
                    sent_idx_list.extend(unk_vec)
            else:
                break

        sent_idx_list.extend([[0]*self.config.numDimensions for i in range(max(self.config.maxSeqLength-len(sent_idx_list), 0))])

        return sent_str_list, sent_idx_list

    def contextualize(self, raw_data, vocab_str=None, vocab_vec=None, save='final_data', overide=None):
        ''' read in raw_data and prepare final data
        :param raw_data: output of read_data, pandas data_frame
        :param overide: default None, it not none, input a pickle file name
        :return: final data contains (labels,sent_str_list,sent_idx_list)
        '''
        contextualize_start_time = time.time()
        if overide:
            data = pickle.load(open("datas/%s" % (overide), "rb"))
            print('finish load final_data from pickle!')
            return data
        else:
            print('Begin prepare for final data...')
            final_data = []
            i = 0
            tmp_seq_len_list=[]
            for index, each in raw_data.iterrows():
                id, list_content, labels, one_hot = each
                list_content = self.clean.sent_num_mask(one_sent=list_content)
                one_hot = eval(one_hot)

                sent_str_list, sent_idx_list = self.sentence2sequence(list_content, vocab_str, vocab_vec)
                final_data.append((one_hot, sent_str_list, sent_idx_list))
                tmp_seq_len_list.append(len(sent_str_list))
                if i % 50 == 0:
                    print(i, tmp_seq_len_list, len(sent_idx_list[0]), 'process sms message in sentence2sequence!')
                    tmp_seq_len_list=[]
                i += 1
            pickle.dump(final_data, open("datas/%s-%s.p" % (save, datetime.now().strftime("%m%d")), "wb+"))

            contextualize_finish_time = time.time()
            spend_time = float((contextualize_finish_time - contextualize_start_time) / 60)
            print("======== total spend for contextualize final data:", spend_time, 'minutes ========\n')

            return np.asarray(final_data)

    def _split_data(self, data, test_size=0.2,random_state=42):
        """ split data use model: sklearn.model_selection
        """
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data
    def split_final(self, final_data, train_size=0, overide=None):
        '''split data set into train, cross validation, test.

        :param final_data: (id,raw_content, scene_category), None when overide is True
        :param train_size: if = 0 , means use full length of training. if = 0.9, means only use 10% of training set.
        :param overide: False or None if not overide; a time with month and date(eg.0801) if overide to get that date's data.
        :return: seperate data into train, dev, test
        '''

        if overide:
            train_data = pickle.load(open("datas/train_data%s.p" % (overide), "rb"))
            cv_data = pickle.load(open("datas/cv_data%s.p" % (overide), "rb"))
            test_data = pickle.load(open("datas/test_data%s.p" % (overide), "rb"))

            print('finish load split data from pickle!')
            return train_data, cv_data, test_data
        else:
            train_dev, test_data = train_test_split(final_data, test_size=0.2, random_state=42)
            train_data, cv_data = train_test_split(train_dev, test_size=0.25, random_state=42)
            train_data, _ = train_test_split(train_data, test_size=train_size, random_state=42)

            today = datetime.now().strftime("%m%d")

            pickle.dump(train_data, open("datas/train_data%s.p" % (today), "wb+"))
            pickle.dump(cv_data, open("datas/cv_data%s.p" % (today), "wb+"))
            pickle.dump(test_data, open("datas/test_data%s.p" % (today), "wb+"))

            print('finish split data!')
            return train_data, cv_data, test_data

    def turn_list_into_nparray(self, list):
        """return an array instead of a list"""
        np_list = np.asarray(list)

        return np_list

    def convert_batch_into_array(self, batch):
        """convert list of batch into array"""
        new_batch = []
        for each in batch:
            new_batch.append(self.turn_list_into_nparray(each))
        return np.asarray(new_batch)
    def _add_padding(self,sequences,pad_tok, max_length):

        pad_labels, seq_lengths = [], []

        for sent in sequences:
            pad = max_length - len(sent)
            pad_label = np.append(sent, [pad_tok] * max(pad, 0))

            pad_labels.append(pad_label)
            seq_lengths.append(len(sent))

        return pad_labels, seq_lengths
    def add_padding(self, batch_size_label,pad_tok=0):
        """ padding batch to shape

        :param batch_size_label: shape: [ batch size, seq_length] one batch of labels, each label has the size of[seq_length]
        :param shape: maxSeqLength
        :return:
        """
        # max_length = max(map(lambda x: len(x), batch_size_label))
        max_length = 95
        seuence_padded, sequence_length=self._add_padding(batch_size_label, pad_tok, max_length)

        return seuence_padded, sequence_length


    def store_feed_dict(self,nextBatchVec,nextBatchLabels):
        """update self.feed_dict and self.seq_lengths
        """
        nextBatchLabels = self.convert_batch_into_array(nextBatchLabels)
        nextBatchLabels, seq_lengths = self.add_padding(nextBatchLabels)
        feed_dict = {self.batch_vec_data: nextBatchVec, self.labels: nextBatchLabels,
                     self.sequence_lengths: seq_lengths}
        return feed_dict, seq_lengths




    def add_placeholders(self):
        # self.config.batchSize, Sequence Length
        self.labels = tf.placeholder(tf.int32, [None, None], name='labels')  # self.config.batchSize


        self.batch_vec_data = tf.placeholder(
            tf.float32, [None, None, self.config.numDimensions],
            "batch_vec_data")  # batchSize # self.config.batchSize

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")

    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)
            lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=self.config.drop_out)

            lstmCell_bw = tf.contrib.rnn.BasicLSTMCell(self.config.lstmUnits)
            lstmCell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_bw, output_keep_prob=self.config.drop_out)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(lstmCell_fw,
                                                                        lstmCell_bw, self.batch_vec_data,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            value = tf.concat([output_fw, output_bw], axis=-1)

        with tf.variable_scope("proj"):
            weight = tf.Variable(tf.truncated_normal([2 * self.config.lstmUnits, self.config.numClasses]),
                                 name='weight')
            bias = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]), name='bias')

            max_seq_len = tf.shape(value)[1]

            value = tf.reshape(value, [-1, 2 * self.config.lstmUnits])
            predict = tf.matmul(value, weight) + bias
            # logits
            self.prediction = tf.reshape(predict,
                                         [-1, max_seq_len, self.config.numClasses])  # batchSize

    def add_loss_op(self):
        with tf.variable_scope("pred"):
            if self.config.use_crf:

                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.prediction, self.labels, self.sequence_lengths)
                self.trans_params = trans_params  # need to evaluate it for decoding
                self.loss = tf.reduce_mean(-log_likelihood)
            else:
                # accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
                # loss = tf.reduce_mean(
                #     tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
                #                      # + self.config.beta * tf.nn.l2_loss(weight) + self.config.beta * tf.nn.l2_loss(bias)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.prediction, labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate, beta1=0.9,
                                                    beta2=0.999).minimize(self.loss)

    def predict_batch(self, feed_dict, seq_lengths):

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                [self.prediction, self.trans_params], feed_dict=feed_dict)
            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, seq_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences
        else:
            labels_pred = tf.cast(tf.argmax(self.prediction, axis=-1),
                                       tf.int32)
            labels_pred = self.sess.run(labels_pred, feed_dict=feed_dict)
            return labels_pred



    def evaluate(self, cv_data, out_file=None):
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.

        labels, sent_str_list, sent_idx_list = zip(*cv_data)

        feed_dict, seq_lengths = self.store_feed_dict(sent_idx_list, labels)

        labels_pred = self.predict_batch(feed_dict, seq_lengths)

        predict_out = []

        for lab, lab_pred, length, sent in zip(labels, labels_pred, seq_lengths, sent_str_list):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            accs += [a == b for (a, b) in zip(lab, lab_pred)]
            tag_dict = self.config.get_class_dict()
            lab_chunks = set(self.clean.get_chunks(lab, tag_dict))
            lab_pred_chunks = set(self.clean.get_chunks(lab_pred,
                                                        tag_dict))
            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)

            if out_file:
                pred_out = self.clean.output_predict_result(sent, lab_pred_chunks)
                predict_out.append(pred_out)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)


        if out_file:
            with open(out_file,'w') as outFile:
                writer = csv.writer(outFile)

                for result in predict_out:
                    writer.writerow(result)

        return {"acc": 100 * acc, "f1": 100 * f1, "precision": 100 * p, "recall": 100 * r }

    def init_train(self):
        tf.reset_default_graph()

        self.add_placeholders()
        self.add_logits_op()
        self.add_loss_op()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver()
        # s_w = tf.summary.FileWriter('tensorboard',sess.graph)

    def train_(self, train_data, cv_data, save=None):
        # for cross validation
        minibatch_cv, _ = self._split_data(cv_data, test_size=0.)
        train_start_time = time.time()
        # begin training
        for i in range(self.config.iterations):
            batch = np.random.randint(len(train_data), size=self.config.batchSize)
            batch_data = [train_data[k] for k in batch]

            nextBatchLabels, _, nextBatchVec = zip(
                *batch_data)  # zip(*batch_data) returns labels, sent_str_list, sent_idx_list

            feed_dict, _ =self.store_feed_dict(nextBatchVec, nextBatchLabels) # initialize self.feed dict

            _, loss_ = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)



            # calculate accuracy
            if (i + 1) % 20 == 0:
                if (i+1) == self.config.iterations:
                    metrics = self.evaluate(cv_data, out_file="outputs/predict_ner.csv")
                else:
                    metrics = self.evaluate(cv_data)
                metrics['loss'] = round(loss_, 6)

                msg = " - ".join(["{} {:04.2f}".format(k, v)
                                  for k, v in metrics.items()])
                print("Batch %s: \n%s " % (i + 1, msg))



        train_finish_time = time.time()
        spend_time = float((train_finish_time - train_start_time) / 60)
        print("======== Total spend for training bi-lstm & crf model:", spend_time, 'minutes ========\n')
            # help made judgment of converging.
            # if 100 - train_accu < 0.25:
            #     convertion += 1

        # if save:
        #     today = datetime.now().strftime("%m%d")
        #     # Save the variables to disk.
        #     save_path = self.saver.save(self.sess, "tf_model/model-%s%s.ckpt" % (save,today))
        #     print("Model saved in path: %s" % save_path)
        #     # print("saved to models/model-%s%s.p" % (save,today))

    def train(self, train_data, cv_data, save=None):
        self.init_train()

        self.train_(train_data=train_data, cv_data=cv_data, save=save)
