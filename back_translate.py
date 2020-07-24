#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/07/24 11:22:34
@Author  :   Zhiyang.zzy 
@Desc    :   
'''
# 来源百度翻译api：http://api.fanyi.baidu.com/api/trans/product/apidoc
# translate包和textblob包

# here put the import lib
import http.client
import hashlib
import urllib
import random
import json
import time

# 调用百度翻译API将中文翻译成英文
def baidu_translate(ori_query: str, toLang='zh', fromLang='auto'):
    """
    ori_query: 原query
    fromLang: 原文语种
    toLang: 译文语种
    return: query_qrr: 翻译语句列表
    来源百度翻译api例子：http://api.fanyi.baidu.com/api/trans/product/apidoc
    """
    appid = '20200724000525424'
    secretKey = 'h3zgcJiXYgpEFzDuWjnM'
    query_arr = []
    # 休息一秒，降低调用频率
    time.sleep(1)
    myurl = '/api/trans/vip/translate'

    salt = random.randint(32768, 65536)
    sign = appid + ori_query + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(ori_query) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        # response是HTTPResponse对象
        response = httpClient.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)
        # print(result)
        for each in result['trans_result']:
            query_arr.append(each['dst'])
        return query_arr

    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()
    return query_arr


if __name__ == '__main__':
    contents = '打开空调'
    # 根据语言列表，可以翻译成多个句子, language_list = ['en', 'jp', 'kor', 'fra']
    out_arr = []
    # lan_list = "en,jp,kor,fra,spa,th,ara,ru,pt,de,it,el,nl,pl,bul,est,dan,fin,cs,rom,slo,swe,hu,cht,vie"
    lan_list = "en,jp,kor".split(",")
    for tmp_lan in lan_list:
        # tmp_arr = baidu_translate(contents, tmp_lan)
        for tmp_q in baidu_translate(contents, tmp_lan):
            out_arr.extend(baidu_translate(tmp_q, 'zh'))
    print(out_arr)
