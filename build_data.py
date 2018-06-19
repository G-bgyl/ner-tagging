from models.data_utils import Clean_data


clean = Clean_data()

# Build data for w2v representation
# data = clean.pd_select_column('datas/msg_.txt',['sms_message_id','raw_mark_split','mark'])
# clean.delete_repetition('datas/msg_.txt',['raw_mark_split'],'datas/msg_all.csv')
clean.turn_type_into_class('datas/msg_all.csv', 'datas/msg_all_cls.csv')



# dt = clean.delete_repetition_by_df(data,['raw_mark_split'],'datas/ner_del_rep.csv')
# clean.turn_type_into_class('datas/ner_del_rep.csv', 'datas/ner_del_rep_cls.csv')