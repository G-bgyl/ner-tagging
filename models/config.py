
class Config():
    def __init__(self):
        #params for build data
        self.maxSeqLength = 80

        # params for tensorflow training
        self.batchSize = 200
        self.lstmUnits = 64
        self.numClasses = 8
        self.iterations = 1000
        self.numDimensions = 300
        self.dropout = 0.75

        self.beta = 0.01

    def param_model(self,nchars,dim_char,char_hidden_size,hidden_size,num_class):
        self.nchars = nchars
        self.dim_char = dim_char
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size
        self.num_class = num_class


