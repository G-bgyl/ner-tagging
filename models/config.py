import pandas as pd
import os

class Config():

    def __init__(self):
        #params for build data
        self.maxSeqLength = 150

        # params for tensorflow training
        self.batchSize = 200
        self.lstmUnits = 64
        self.numClasses = self.cal_numClasses()
        self.iterations = 10000
        self.numDimensions = 300
        self.drop_out = 0.75
        self.use_crf = False
        self.beta = 0.01
        self.learning_rate = 0.0025

    def param_model(self,nchars,dim_char,char_hidden_size,hidden_size,num_class):
        self.nchars = nchars
        self.dim_char = dim_char
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size
        self.num_class = num_class

    def cal_numClasses(self, configFile='/datas/ner_type.txt'):
        data = pd.read_csv(os.getcwd()+configFile, sep="\t")
        return data.shape[0]

