import pandas as pd
from .general_utils import cut_text, sqlDataFame, msg_postag

#get data
sql = "select id, raw_content, split_result, mark_split_result, mark_result from sms_messages where scene_code='S0002' and id > 1278;"
df = sqlDataFame(sql, columns=["id", "raw_content", "split_result", "mark_split_result", "mark_result"])
df.dropna(inplace=True)
# get raw content mark split result
cuts = []
for i in df["mark_split_result"]:
      mark_split = eval(i)
      cut_index = [0]
      for j in mark_split:
            cut_index.append(cut_index[-1]+len(j))
      cuts.append(cut_index)
raw_mark_split = [cut_text(j, i) for i, j in zip(cuts, df["raw_content"])]
df["raw_mark_split"] = raw_mark_split
mark_code = {
                'E0001': '短信发送机构',
                'E0012': '账户类型',
                'E0002': '用户账户号',
                'E0003': '交易时间',
                'E0004': '交易对象',
                'E0005': '交易类型',
                'E0006': '交易金额',
                'E0007': '余额类型',
                'E0008': '账户余额'
                }
# get message mark split mark result
sms_marks = []
for i,j in zip(df["raw_mark_split"], df['mark_result']):
    raw_mark_split = i
    mark_result = eval(j)
    mark_result = [(k[1], k[0]) for k in mark_result if k[1] != '']
    mark_result = dict(mark_result)
    mark = []
    for l in range(len(raw_mark_split)):
        if l in mark_result.keys():
            mark.append(mark_result[l])
        else:
            mark.append("O")

    sms_marks.append(mark)

df["sms_mark"] = sms_marks

# get message split result mark result
split_mark = []
for i in df.index:
    mark_split = df.loc[i, 'raw_mark_split']
    split_result = eval(df.loc[i, 'split_result'])
    mark = df.loc[i, 'sms_mark']
    mark_split_index = 0
    split_result_index = 0
    m = 0
    n = 0
    mark_list = []
    while n < len(split_result):
        mark_split_index += len(mark_split[m])
        split_result_index += len(split_result[n])
        if mark_split_index == split_result_index:
            mark_list.append(mark[m]+"-S")
            m += 1
            n += 1
        else:
            M = 0
            while mark_split_index != split_result_index:
                # if mark_split_index-len(mark_split[m]) == split_result_index - len(split_result[n]):
                if M == 0:
                    mark_list.append(mark[m] + "-B")
                    M += 1
                else:
                    mark_list.append(mark[m] + "-M")
                n += 1
                split_result_index += len(split_result[n])
            mark_list.append(mark[m]+"-E")
            m += 1
            n += 1
    split_mark.append(mark_list)

df['split_mark'] = split_mark

# get raw content split result
cuts = []
for i in df["split_result"]:
      mark_split = eval(i)
      cut_index = [0]
      for j in mark_split:
            cut_index.append(cut_index[-1]+len(j))
      cuts.append(cut_index)

raw_split = []
for i,j in zip(cuts, df["raw_content"]):
      raw_split.append(cut_text(j, i))
df["raw_split"] = raw_split

# get raw content split result postag
raw_split_postag = [msg_postag(i) for i in df["raw_split"]]
df["raw_split_postag"] = raw_split_postag


for i in df.index:
    if "E0007-M" in df.loc[i, "split_mark"]:
        print("------------------------------------")
        print(df.loc[i, "id"])
        print(df.loc[i, "raw_split"])
        print(df.loc[i, "raw_mark_split"])
        print(df.loc[i, "split_mark"])
        d = dict(zip(df.loc[i, 'split_mark'], df.loc[i, 'raw_split']))
        print(d['E0007-M'])
# df.to_csv("./Data/msg_mark.txt",sep= "\t", index=False)