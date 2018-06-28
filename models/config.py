import pandas as pd
import os

class Config():

    def __init__(self):
        self.tagFile = os.getcwd() + '/datas/ner_type.txt'

        #params for build data
        self.maxSeqLength = 95

        # params for tensorflow training
        self.batchSize = 200
        self.lstmUnits = 256
        self.numClasses = self.cal_numClasses()
        self.iterations = 600
        self.numDimensions = 1024
        self.drop_out = 0.75
        self.use_crf = True
        self.learning_rate = 0.0025



    def param_model(self,nchars,dim_char,char_hidden_size,hidden_size,num_class):
        self.nchars = nchars
        self.dim_char = dim_char
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size
        self.num_class = num_class

    def cal_numClasses(self):
        data = pd.read_csv(self.tagFile, sep="\t")
        return data.shape[0]

    def get_class_dict(self):
        tag_dict={}
        data = pd.read_csv(self.tagFile, sep="\t")

        for _,(tag,id) in data.iterrows():
            tag_dict[tag] = id
        return tag_dict


