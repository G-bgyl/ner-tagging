'''
    Entrance for train a word2vec or char2vec vocab.
'''
from models.w2v_model import Word2Vec

def main(col_namList, char_embedding,dimension,resource_file,out_file):
    """ output word embeddings
    args:
        file_word_seg: processed data from split_clean.py
    """
    w2v=Word2Vec()

    sentences = w2v.clean_w2v_data(resource_file, col_name_list = col_namList, out_file=None, char=char_embedding)  # out_file='datas/raw_split_for_w2v.csv'
    w2v.init_model(sentences,size=dimension)
    w2v.train_w2v(sentences)
    w2v.output_w2v(path='datas/',file=out_file)
    if not char_embedding:
        w2v.evaluate_w2v()

if __name__ == '__main__':
    # for word embedding
    col_namList=['raw_split']
    char_embedding=False
    dimension = 100
    resource_file = 'datas/sms1w_wdSeg_real.txt'
    out_file = ['vocab_str_train_100_unmask.p', 'vocab_vec_train_100_unmask.p']

    # # for char embedding
    # col_namList = ['raw_split']
    # char_embedding = True
    # dimension = 100
    # resource_file = 'datas/sms1w_wdSeg_real.txt'
    # out_file = ['char_str_train_100_unmask.p', 'char_vec_train_100_unmask.p']

    main(col_namList, char_embedding, dimension, resource_file, out_file)