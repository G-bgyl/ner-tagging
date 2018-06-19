import tensorflow as tf
import numpy as np

import pickle
import time



from .config import Config


class model():

    def __init(self):
        # get from pretrained embeddings
        embeddings = {}
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None])

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None])
        L = tf.Variable(embeddings, dtype=tf.float32, trainable=False)



    # config: see config.py

    # -------------
    # Dealing with Unknown words in zh_w2v
    # -------------

    def generate_new(unk, vocab_str, vocab_vec, Gmean, Gvar):
        """
          Input: unknown word
          Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
        """
        # global vocab_str,vocab_vec, Gmean, Gvar

        RS = np.random.RandomState()
        # append new word string into vocab_str
        vocab_str.append(unk)
        # create new word vector based on normal distribution into vocab_vec
        vocab_vec.append(RS.multivariate_normal(Gmean, np.diag(Gvar)))

        return vocab_vec[-1]



    # ----------------------------------
    # turn one sentence into list of index
    # ----------------------------------
    def sentence2sequence(self,sentence, vocab_str, vocab_vec, Gmean, Gvar):
        """
            - Turns an input paragraph into an (m,d) matrix,
                where n is the number of tokens in the sentence
                and d is the number of dimensions each word vector has.
              Input: sentence, string
              Output: list of index of each word segment in sentence
        """
        sent_raw_list = sentence
        sent_str_list = []
        sent_idx_list = []

        for j in range(len(sent_raw_list)):
            if len(sent_idx_list) < Config.maxSeqLength:
                word = sent_raw_list[j]
                try:
                    sent_idx_list.append(vocab_vec[vocab_str.index(word)])
                    sent_str_list.append(word)
                except ValueError:
                    # Greedy search for word: if it did not find the whole word in zh_w2v, then truncate tail to find
                    i = len(word)
                    # np.average(data, axis=0)
                    sent_idx_list.append([])
                    sent_str_list.append([])
                    while len(word) > 0:
                        wd = word[:i]  # shallow copy
                        if wd in vocab_str:
                            # Make sure the length of th list is less than maxSeqLength
                            if len(sent_idx_list) < Config.maxSeqLength:
                                sent_idx_list.append(vocab_vec[vocab_str.index(wd)])
                                sent_str_list.append(wd)
                                word = word[i:]  # set to whatever the rest of words is, eg. hallway --> hall, way, split to 2 words
                                i = len(word)
                                continue
                            else:
                                break
                        else:
                            i = i - 1
                        if i == 0:
                            # Make sure the length of th list is less than maxSeqLength
                            if len(sent_idx_list) < Config.maxSeqLength:
                                sent_idx_list.append(self.generate_new(word, vocab_str, vocab_vec, Gmean, Gvar))
                                sent_str_list.append(word)
                                break
                            else:
                                break
            else:
                break
        # make sure the length of two return params are fixed.
        # Padding
        left_len = Config.maxSeqLength - len(sent_idx_list)
        if left_len > 0:
            sent_idx_list.extend(np.zeros((300,)) for i in range(left_len))
            sent_str_list.extend([''] * left_len)
        return sent_str_list, sent_idx_list

    # -----------------------------
    # read in and prepare data
    # -----------------------------
    def contextualize(self,raw_data, vocab_str=None, vocab_vec=None, Gmean=None, Gvar=None, overide=False):
        '''
        :param raw_data: output of read_data
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
            for each in raw_data:
                string_content, labels = each
                sent_str_list, sent_idx_list = self.sentence2sequence(string_content, vocab_str, vocab_vec, Gmean, Gvar)
                final_data.append((labels, sent_str_list, sent_idx_list))
                if i % 500 == 0:
                    print(i, len(sent_str_list), len(sent_idx_list[0]), 'process sms message in sentence2sequence!')
                i += 1
            pickle.dump(final_data, open("DATA/final_data_clean0525.p", "wb+"))

            contextualize_finish_time = time.time()
            spend_time = float((contextualize_finish_time - contextualize_start_time) / 60)
            print("======== total spend for contextualize final data:", spend_time, 'minutes ========\n')

            return np.asarray(final_data)












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
