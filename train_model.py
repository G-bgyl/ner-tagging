from models.data_utils import Clean_data
from models.config import Config
from models.ner_crf import Model


def main(file_name):

    clean = Clean_data()
    vocab_str = clean.read_pickle('datas/vocab_str.p')
    vocab_vec = clean.read_pickle('datas/vocab_vec.p')
    df = clean.read_file(file_name)

    config = Config()
    model = Model(config)

    # final_data_all = model.contextualize(raw_data=df, vocab_str=vocab_str, vocab_vec=vocab_vec)
    final_data_all = model.contextualize(raw_data=None, overide='train_data-0621.p')

    train_data, cv_data, test_data = model.split_final(final_data_all)
    model.train(train_data,'try')


if __name__ == "__main__":
    file = 'datas/ner_del_rep_cls_1h.csv'
    main(file)
