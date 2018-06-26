
from models.w2v_model import Word2Vec

def main():
    w2v=Word2Vec()

    file_word_seg = 'datas/sms1w_wdSeg_real.txt'
    sentences = w2v.clean_w2v_data(file_word_seg)
    w2v.init_model(sentences)
    w2v.train_w2v(sentences)
    w2v.output_w2v(path='datas/',file=['vocab_str_train.p','vocab_vec_train.p'])
    w2v.evaluate_w2v()

if __name__ == '__main__':
    main()