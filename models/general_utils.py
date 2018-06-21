import pymysql

import re

conn = pymysql.Connect(
    user='root',
    passwd='root',
    db='vine',
    charset='utf8'
)


def sqlDataFame(sql, columns=None):
    cursor = conn.cursor()
    cursor.execute(sql)
    if columns is None:
        result = pd.DataFrame(list(cursor.fetchall()))
    else:
        result = pd.DataFrame(list(cursor.fetchall()), columns=columns)
    return result

def cut_text(text, cuts):
    left = 0
    result = []
    cuts += [len(text)]
    cuts.sort()
    for c in cuts:
        tmp = text[left:c]
        if tmp:
            result += [tmp]
            left = c
    return result

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
    regex = "(\d+年\d+月\d+日)|(\d+年\d+月)|(\d+月\d+日)|((?<![-\d])\d+日)|(([012]{2})?\d{2}-\d{1,2}-\d{1,2})|([01]?\d-[0123]?\d(?!\d))|2018\d{4}|(\d+/\d+/\d+)|(\d+/\d+)|\d+-\d+ 月|[0,1]\d[0,1,2,3]\d"
    regex_time = "(\d+:\d+:\d+)|(\d+:\d+)|(\d+时\d+分\d+秒)|(\d+时\d+分)|(\d+时)|(\d+：\d+)|(\d{4} \d{4})|\d{1,2}日[0,1,2]\d点\d{2}分"
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