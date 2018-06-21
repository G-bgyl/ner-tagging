
import re
from .general_utils import sqlDataFame

def is_Chinese_word(string):
    flag = False
    if re.search('[\u4e00-\u9fa5]', string) is not None:
        flag=True
    return flag

def is_string(string):
    flag = False
    if (re.search('[\u4e00-\u9fa5]', string) is None) and (re.search('[a-zA-Z]', string) is not None):
        flag = True
    return flag

def is_digit(string):
    flag = False
    if (not is_Chinese_word(string)) and (not is_string(string)) and (re.search("\d", string) is not None):
        flag = True
    return flag

def is_time(string):
    regex = "(\d+年\d+月\d+日)|(\d+年\d+月)|(\d+月\d+日)|((?<![-\d])\d+日)|(([012]{2})?\d{2}-\d{1,2}-\d{1,2})|([01]?\d-[0123]?\d(?!\d))|2018\d{4}|(\d+/\d+/\d+)|(\d+/\d+)|\d+-\d+ 月"
    regex_time = "(\d+:\d+:\d+)|(\d+:\d+)|(\d+时\d+分\d+秒)|(\d+时\d+分)|(\d+时)|(\d+：\d+)|(\d{4} \d{4})"
    # match_dt日期匹配，match_t时间匹配
    match_dt = re.search("(" + regex + ")" + "[\s,，]?" + "(" + regex_time + ")?", string)
    if match_dt is not None:
        return True
    else:
        return False

def msg_postag(msg_split):
    postag_list = []
    for i in msg_split:
        if is_Chinese_word(i):
            postag_list.append("c")
        elif is_digit(i):
            postag_list.append("d")
        elif is_time(i):
            postag_list.append("t")
        elif is_string(i):
            postag_list.append("s")
        else:
            postag_list.append("o")
    return postag_list

sql = "select id, raw_content, split_result from sms_messages where scene_code='S0002';"

df = sqlDataFame(sql, columns=["sms_message_id", "raw_content", "split_result"])
df.dropna(inplace=True)
postag = [msg_postag(eval(i)) for i in df["split_result"]]
df["postag"] = postag
df.to_csv("./Data/msg_postag.txt", sep="\t", index=False)

