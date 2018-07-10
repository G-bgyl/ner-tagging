import pandas as pd
import numpy as np
import pickle
import re

from models.general_utils import is_digit, is_time, is_string

NONE = "O-S"

class Clean_data():

    def __init__(self):
        pass
        self.ner_type = {  'E0001':'短信发送机构',
                            'E0002':'用户账户号',
                            'E0003':'交易时间',
                            'E0004':'交易对象',
                            'E0005':'交易类型',
                            'E0006':'交易金额',
                            'E0007':'账户余额',
                            'E0008':'余额类型',
                            'E0012':'账户类型'
                            }
        # self.mark_class = { 'E0001': 1,
        #                     'E0002': 2,
        #                     'E0003': 3,
        #                     'E0004': 4,
        #                     'E0005': 5,
        #                     'E0006': 6,
        #                     'E0007': 7,
        #                     'E0008': 8,
        #                     'E0012': 9,
        #                     'O':0
        #                    }

    def read_pickle(self,fileName):
        """ load in a pickle file

                       :param file_name: txt file
                       :return: a dataframe
                       """
        print('Reading pickle...')
        data = pickle.load(open("%s" % (fileName), "rb"))
        return data

    def read_file(self,fileName):
        """ read in a csv/txt file

                :param file_name: txt file
                :return: a dataframe
                """
        print('Reading data...')
        data = pd.read_csv(fileName, sep="\t")
        print('Successfully reading data (Clean_data.read_file)')
        return data


    def pd_select_column(self, file_name, col_name_list, out_file=None):
        """ select column from a txt/csv file

        :param file_name: txt file
        :param col_name_list: a list of column names that needed
        :return: a dataframe
        """
        print('Reading data...')
        data = pd.read_csv(file_name, sep="\t")
        df = data[col_name_list]

        # output a file if needed
        if out_file:
            df.to_csv(out_file,sep='\t', header=True ,index=False)
            print('Dataframe\'s shape:{}' .format (df.shape))
            print('Output file: %s\n'%(out_file))
        else:
            print('Dataframe\'s shape:{}\n' .format (df.shape))

        return df



    def delete_repetition(self, file_name, subset, out_file=None):
        """

        :param file_name: string of origin file name
        :param subset: a sequence or a list of column name that become the standard of defining repetition
        :param out_file:  string of output file name
        :return: a pandas dataframe
        """

        print('Reading data...')
        data = pd.read_csv(file_name, sep="\t")
        df = data.drop_duplicates(subset=subset)
        # output a file if needed
        if out_file:
            df.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}' .format(df.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n' .format(df.shape))

        return df

    def delete_repetition_by_df(self, data, subset, out_file=None):
        """

        :param file_name: string of origin file name
        :param subset: a sequence or a list of column name that become the standard of defining repetition
        :param out_file:  string of output file name
        :return: a pandas dataframe
        """

        print('Reading DataFrame...')
        df = data.drop_duplicates(subset=subset)
        # output a file if needed
        if out_file:
            df.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}' .format(df.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n' .format(df.shape))

        return df

    def turn_type_into_class(self, file_name, input_col, out_col, out_file=None, saveConfig= 'datas/ner_type.txt'):
        """turn type(string) into class(number)

        :param data: a file name
        :return: df
        """

        data = pd.read_csv(file_name, sep="\t")
        gold_std_cls_list = []
        self.mark_class = {}
        for index, item in data.iterrows():
            gold_std_type = eval(item[input_col])
            gold_std_cls = []
            for id, each in enumerate(gold_std_type):
                if each in self.mark_class:
                    gold_std_cls.append(self.mark_class[each])
                else:
                    self.mark_class[each]=len(self.mark_class)
                    gold_std_cls.append(self.mark_class[each])
            gold_std_cls_list.append(gold_std_cls)
        data[out_col] = gold_std_cls_list

        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}'.format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n'.format(data.shape))


        if saveConfig:
            dfClass=pd.DataFrame.from_dict(self.mark_class, orient='index')
            dfClass.to_csv(saveConfig, sep='\t', header=False)
        return data
    '''
    def turn_type_into_class_by_df(self, data, input_col, out_col, out_file=None, saveConfig='datas/ner_type.txt'):
        """turn type(string) into class(number)

        :param data: a dataframe
        :return: df
        """

        gold_std_cls_list = []
        self.mark_class = {}
        for index, item in data.iterrows():
            gold_std_type = eval(item[input_col])
            gold_std_cls = []
            for id, each in enumerate(gold_std_type):
                if each in self.mark_class:
                    gold_std_cls.append(self.mark_class[each])
                else:
                    self.mark_class[each] = len(self.mark_class)
                    gold_std_cls.append(self.mark_class[each])
            gold_std_cls_list.append(gold_std_cls)
        data[out_col] = gold_std_cls_list

        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}' .format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n' .format(data.shape))

        if saveConfig:
            dfClass=pd.DataFrame.from_dict(self.mark_class, orient='index')
            dfClass.to_csv(saveConfig, sep='\t', header=True)

        return data'''

    def _turn_cls_into_1hotvc(self,class_,item):
        """turn one number into a vector, length equals to number of types"""
        gold_std_cls = np.array(eval(item[class_]))
        gold_std_cls_1hot = np.zeros((gold_std_cls.size, len(self.mark_class)))
        gold_std_cls_1hot[np.arange(gold_std_cls.size), gold_std_cls] = 1
        return  gold_std_cls_1hot.tolist()

    def turn_cls_into_1hotvec(self, file_name, col_list, out_file = None):
        """turn a dense vector(for ner tagging) into one hot vector

        :param file_name:
        :param col_list:
        :param out_file:
        :return:
        """
        data = pd.read_csv(file_name, sep="\t")

        # main code
        for row in data.iterrows():
            index, item = row
            for class_ in col_list:
                data.set_value(index, class_, self._turn_cls_into_1hotvc(class_,item))

        # for output
        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}'.format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n'.format(data.shape))
        return data

    def turn_list_into_str(self, list_of_list, sep=''):
        """turn list of list(WORD SEGMENTS) into list of string"""
        list_sentences=[]
        for list_of_seg in list_of_list:
            list_sentences.append(sep.join(list_of_seg))
        return list_sentences

    def dataset_num_mask(self, word_seg_list, mask = True):
        """create a mask for numbers, turn time to 8, turn money and others to 1

        :param word_seg_list: a list of list, each sublist contains word segments of a sentece.
        :param mask:
        :return:
        """
        mask_seg_list = []
        for list_ in word_seg_list:
            one_sent = eval(list_[0])
            if mask:
                sentence = self.sent_num_mask(one_sent)
            else:
                sentence = one_sent

            mask_seg_list.append(sentence)
        print('successfully add mask of num!')
        return mask_seg_list

    def sent_num_mask(self, one_sent, mask = True):
        """create a mask for numbers, turn time to 8, turn money and others to 1

        :param sentence: a list of word segments in one sentence
        :return:
        """
        if type(one_sent)== str:  # when the input is a string, like: '['w1','w2','w3'...]'
            one_sent=eval(one_sent)
        elif type(one_sent) == list and len(one_sent)==1:
            # when the input is a list of only one long string(of a list), like:["['w1','w2','w3'...]"]
            one_sent = eval(one_sent[0])

        if mask:
            sentence = []
            for seg in one_sent:
                if is_time(seg):
                    seg = re.sub('\d', '1', seg)
                elif is_digit(seg):
                    seg = re.sub('\d', '8', seg)
                elif is_string(seg):
                    seg = re.sub('[a-zA-Z]','x',seg)
                sentence.append(seg)
        else:
            sentence = one_sent

        return sentence

    def get_chunk_type(self, tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}

        Returns:
            tuple: "B", "PER"

        """
        tag_name = idx_to_tag[tok]
        tag_type = tag_name.split('-')[0]
        tag_class = tag_name.split('-')[-1]
        return tag_class, tag_type

    def get_chunks(self, seq, tags):
        """Given a sequence of tags, group entities and their position

            Args:
                seq: [4, 4, 0, 0, ...] sequence of labels
                tags: dict["O"] = 4

            Returns:
                list of (chunk_type, chunk_start, chunk_end)

            Example:
                seq = [4, 5, 0, 3]
                tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
                result = [("PER", 0, 2), ("LOC", 3, 4)]

            """
        default = tags[NONE]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def output_predict_result(self, sequence, lab_pred_chunks):
        predict_result = ['00'+''.join(sequence)]

        for (n_type,begin,end) in lab_pred_chunks:
            predict_result.append('%s%s: %s' % (n_type[-2:], self.ner_type[n_type], ''.join(sequence[begin:end])))

        predict_result = sorted(predict_result)

        return predict_result