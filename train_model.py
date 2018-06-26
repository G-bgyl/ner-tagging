from models.data_utils import Clean_data
from models.config import Config
from models.ner_crf import Model


def main(file_name):

    clean = Clean_data()
    # vocab_str = clean.read_pickle('datas/vocab_str_train.p')
    # vocab_vec = clean.read_pickle('datas/vocab_vec_train.p')
    df = clean.read_file(file_name)

    config = Config()
    model = Model(config)
    '''don't overide final data'''
    # final_data_all = model.contextualize(raw_data=df, vocab_str=vocab_str, vocab_vec=vocab_vec)
    # train_data, cv_data, test_data = model.split_final(final_data=final_data_all)
    '''overide final data'''
    # final_data_all = model.contextualize(raw_data=None, overide='final_data-0626.p')
    train_data, cv_data, test_data = model.split_final(final_data=None, overide='0626')
    model.train(train_data, 'try')


if __name__ == "__main__":
    file = 'datas/ner_del_rep_cls_1h.csv'
    main(file)
