from models.data_utils import Clean_data

clean = Clean_data()
# data = clean.pd_select_column('datas/msg_.txt',['sms_message_id','raw_mark_split','mark'],'datas/msg_all.csv')

# Build data for w2v representation
# clean.turn_type_into_class('datas/msg_all.csv', 'datas/msg_all_cls.csv')
clean.turn_cls_into_1hotvec('datas/msg_all_cls.csv','datas/msg_all_cls_1h.csv')
# Build data that does not has repetitions, for training
# clean.delete_repetition_by_df(data,['raw_mark_split'],'datas/ner_del_rep.csv')
# clean.turn_type_into_class('datas/ner_del_rep.csv', 'datas/ner_del_rep_cls.csv')
clean.turn_cls_into_1hotvec('datas/ner_del_rep_cls.csv','datas/ner_del_rep_cls_1h.csv')