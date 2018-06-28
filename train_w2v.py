
from models.w2v_model import Word2Vec

def main():
    """
    args:

        file_word_seg: processed data from split_clean.py
    """
    w2v=Word2Vec()

    file_word_seg = 'datas/sms1w_wdSeg_real.txt'
    sentences = w2v.clean_w2v_data(file_word_seg,col_name_list = ['raw_split'], out_file='datas/raw_split_for_w2v.csv')
    w2v.init_model(sentences,size=1024)
    w2v.train_w2v(sentences)
    w2v.output_w2v(path='datas/',file=['vocab_str_train_1024.p','vocab_vec_train_1024.p'])
    w2v.evaluate_w2v()

if __name__ == '__main__':
    main()