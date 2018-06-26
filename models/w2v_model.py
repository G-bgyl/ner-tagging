
import pickle




from models.data_utils import Clean_data

import gensim


class Word2Vec():
    def __init__(self):
        self.clean = Clean_data()

    def clean_w2v_data(self,file_word_seg):


        # clean.pd_select_column returns a dataframe, change it to list


        col_name_list = ['raw_split']
        word_seg_list = self.clean.pd_select_column(file_name=file_word_seg, col_name_list=col_name_list,
                                               out_file='datas/word2vec.csv').values.tolist()


        documents = self.clean.dataset_num_mask(word_seg_list)
        # sentences = self.clean.turn_list_into_str(documents)
        # sentences = gensim.models.word2vec.LineSentence(sentences)
        return documents

    def init_model(self,sentences,size=300,
            window=2,
            min_count=10,
            workers=10,
            sg=1):
        """init gensim model"""
        self.model = gensim.models.Word2Vec(
            sentences,
            size=size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg)

    def train_w2v(self,sentences):
        """train gensim model for word2vec"""
        self.model.train(sentences, total_examples=len(sentences), epochs=10)
        return self.model

    def output_w2v(self,path=None, file = None):
        """ return two lists: string of word and list of word.

        :param model: Instance of Gensim model
        :param path: Relative path of datas directory. Slightly changes each time
        :param file: a list of two str, name of vocab_str and vocab_vec
        :return: 2 lists, vocab_str and vocab_vec
        """
        vocab = self.model.wv.vocab
        vocab_str = list()
        vocab_vec = list()
        for word in vocab:
            vocab_str.append(word)
            vocab_vec.append(self.model.wv[word])
            # print(word, self.model.wv[word],self.model.wv.vocab[word])
        if file:
            pickle.dump(vocab_str, open("%s%s"%(path,file[0]), "wb+"))
            pickle.dump(vocab_vec, open("%s%s"%(path,file[1]), "wb+"))
            print('sample of vocab_vec:{}'.format(vocab_vec[0]))

            print('Successfully output vocab_str & vocab_vec to pickle!')

        return vocab_str, vocab_vec

    def evaluate_w2v(self):
        # evaluate
        print()
        print()
        w1 = '滴滴出行'
        print(w1, self.model.wv.most_similar(positive=w1))
        w1 = '1月11日1时11分'
        print(w1, self.model.wv.most_similar(positive=w1))
        w1 = '中信银行'
        w2 = '平安银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '邮储银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '广发银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '中信银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '华夏银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '北京银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '渤海银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '中国农业银行'
        print(w2, self.model.wv.similarity(w2, w1))
        w2 = '光大银行'
        print(w2, self.model.wv.similarity(w2, w1))
        print(w1, self.model.wv.most_similar(positive=w1))
        w1 = '拼多多'
        print(w1, self.model.wv.most_similar(positive=w1))
        w1 = '88.88'
        print(w1, self.model.wv.most_similar(positive=w1))
        w1 = '跨行汇款'
        print(w1, self.model.wv.most_similar(positive=w1))
        print()
        print(len(self.model.wv['银联']))



if __name__ =='__main__':
    pass
