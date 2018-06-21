import pandas as pd
import numpy as np
import pickle

class Clean_data():

    def __init__(self):
        self.mark_type = {  'E0001':'短信发送机构',
                            'E0002':'用户账户号',
                            'E0003':'交易时间',
                            'E0004':'交易对象',
                            'E0005':'交易类型',
                            'E0006':'交易金额',
                            'E0007':'账户余额',
                            'E0008':'余额类型',
                            'E0012':'账户类型'
                            }
        self.mark_class = { 'E0001': 1,
                            'E0002': 2,
                            'E0003': 3,
                            'E0004': 4,
                            'E0005': 5,
                            'E0006': 6,
                            'E0007': 7,
                            'E0008': 8,
                            'E0012': 9,
                            'O':0
                           }

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

    def turn_type_into_class(self, file_name, out_file=None):
        """turn type(string) into class(number)

        :param data: a file name
        :return: df
        """
        data = pd.read_csv(file_name, sep="\t")
        for index, item in data.iterrows():
            gold_std_type = eval(item['mark'])
            gold_std_cls = []
            for id, each in enumerate(gold_std_type):
                gold_std_cls.append(self.mark_class[each])
            data.set_value(index, 'mark', gold_std_cls)

        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}'.format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n'.format(data.shape))

        return data

    def turn_type_into_class_by_df(self, data, out_file=None):
        """turn type(string) into class(number)

        :param data: a dataframe
        :return: df
        """

        for index, item in data.iterrows():
            gold_std_type = eval(item['mark'])
            gold_std_cls = []
            for id, each in enumerate(gold_std_type):
                gold_std_cls.append(self.mark_class[each])
            data.set_value(index, 'mark', gold_std_cls)

        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}' .format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n' .format(data.shape))

        return data

    def turn_cls_into_1hotvec(self,file_name, out_file = None):
        """turn a dense vector(for ner tagging) into one hot vector"""
        data = pd.read_csv(file_name, sep="\t")

        # main code
        for index, item in data.iterrows():
            gold_std_cls = np.array(eval(item['mark']))
            gold_std_cls_1hot = np.zeros((gold_std_cls.size, len(self.mark_class)))
            gold_std_cls_1hot[np.arange(gold_std_cls.size), gold_std_cls] = 1
            data.set_value(index, 'mark', gold_std_cls_1hot.tolist())

        # for output
        if out_file:
            data.to_csv(out_file, sep='\t', header=True, index=False)
            print('Dataframe\'s shape:{}'.format(data.shape))
            print('Output file: %s\n' % (out_file))
        else:
            print('Dataframe\'s shape:{}\n'.format(data.shape))

        return data