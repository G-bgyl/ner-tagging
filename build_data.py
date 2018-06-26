from models.data_utils import Clean_data

clean = Clean_data()
data = clean.pd_select_column('datas/sms_messages_ner_all_col.txt',['id','raw_split','split_mark'],'datas/msg_all.csv')


# Build data that does not has repetitions, for training
clean.delete_repetition_by_df(data,['raw_split'],'datas/ner_del_rep.csv')

clean.turn_type_into_class(
    file_name='datas/ner_del_rep.csv',
    input_col='split_mark',
    out_col='mark',
    out_file='datas/ner_del_rep_cls.csv')
clean.turn_cls_into_1hotvec('datas/ner_del_rep_cls.csv',
                            'datas/ner_del_rep_cls_1h.csv')