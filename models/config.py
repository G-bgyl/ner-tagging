
class Config():

    #params for build data
    maxSeqLength = 280

    # params for tensorflow training
    batchSize = 200
    lstmUnits = 64
    numClasses = 8
    iterations = 1000
    numDimensions = 300

    def param_model(self,nchars,dim_char,char_hidden_size,hidden_size,num_class):
        self.nchars = nchars
        self.dim_char = dim_char
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size
        self.num_class = num_class


