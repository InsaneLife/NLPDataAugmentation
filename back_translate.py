#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2020/07/23 11:22:34
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
    appid = '20200724000525424' # appid 需要在 https://api.fanyi.baidu.com/ 申请， 请填写您的 appid
    secretKey = open("./data/password").readline().strip() # secretKey 需要在 https://api.fanyi.baidu.com/ 申请，这里我的key脱敏了, 请填写您的key
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

def back_translate(query_arr):
    out_arr = {}
    for query in query_arr:
        cur_arr = []
        # 根据语言列表，可以翻译成多个句子, language: en,jp,kor,fra,spa,th,ara,ru,pt,de,it,el,nl,pl,bul,est,dan,fin,cs,rom,slo,swe,hu,cht,vie...
        lan_list = "en,jp,kor,fra,spa,th,ara,ru,pt,de,it,el,nl,pl,bul,est,dan,fin,cs,rom,slo,swe,hu,cht,vie".split(",")
        for tmp_lan in lan_list:
            for tmp_q in baidu_translate(query, tmp_lan):
                cur_arr.extend(baidu_translate(tmp_q, 'zh'))
        out_arr[query] = list(set(cur_arr))
    return out_arr

def write_translate2result(file_path='./data/input'):
    with open(file_path, encoding='utf-8') as f, open(file_path + "_translate", 'w', encoding='utf-8') as out:
        query_arr = [x.strip() for x in f.readlines()]
        for k, v in back_translate(query_arr).items():
            out.write("\t".join([k, ";".join(v)]) + "\n")



if __name__ == '__main__':
    write_translate2result()
    # result:
    # 帮我查一下航班信息	帮我查一下航班信息;请帮我查一下飞机的情报。;帮我检查航班信息。;检查我的航班信息。;检查航班;查一下我的飞行记录。;查一下我的航班信息。;检查一下飞行资料;检查飞行数据。;帮我查一下航班信息。;帮我查一下VOO信息;帮我查一下航班数据。;帮我查一下飞行记录。;请查一下飞机的信息。;幫我查一下班機資訊
    # 查一下航班信息	打听一下航班的信息。;检查航班;检查VOO信息;查看航班信息;检查飞行数据。;四航班检查;检查航班信息。;航班信息验证;查一下班機資訊;请查一下飞机的信息。;检查飞行信息;检查航班信息
    # 附近有什么好玩的	这里有什么有趣的？;这里有什么有趣的;这个地方有什么有趣的？;这里有什么好玩的？;这里有什么好玩的;这个地方有什么好玩的？;发生什么事了？;这附近有什么好玩的地方吗;有什么有趣的？;附近有什么好玩的吗？;这附近有什么好玩的;附近有什麼好玩的;附近有什么有趣的东西吗？
