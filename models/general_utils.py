import pymysql
import pandas as pd
import re

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

