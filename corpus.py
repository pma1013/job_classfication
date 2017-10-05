# -*-coding: utf-8 -*-
import os
import sys
import re
import urllib.request
import urllib.parse
import json
import codecs
import MeCab
from gensim import corpora, models, matutils

from sklearn.ensemble import RandomForestClassifier

JOB_FILE = "joblist.json"
DICT_FILE = "kyujindict.txt"

# Search Pram
SEARCH_HOST = "localhost"
PORT = "8890"
ROWS = "100"

def make_query(kwd):
    '''
    solrへの検索クエリを作成する
    '''
    # title + job_fieldを検索対象に
    kwd = "((title:" + kwd + ") OR (job_field:" + kwd + "))"
    search_query = urllib.parse.quote(kwd.encode('utf-8'))
    # title+job_fieldを参照
    url = "http://%s:%s/solr/rb_s/select?q=%s&wt=json&indent=true&fl=kyujin_uniqueid,title,job_field,search_field&sort=score%%20desc&rows=%s" % (SEARCH_HOST, PORT, search_query, ROWS)
    
    return url

def search_job(url):
    '''
    Solrに検索クエリを投げる
    '''
    try:
        response_str = urllib.request.urlopen(url).read()
        response_dict = json.loads(response_str.decode('utf-8'))

        return response_dict

    except urllib.error.URLError as e:
        print(e.code)
        print(e.read())
        print(url)
        sys.exit()

def get_label(u_key):
    '''
    unique_keyからlabel返却
    '''
    job_dic = get_job_dic()
    for i in job_dic.keys():
        matchOB = re.match(i, u_key)
        if matchOB:
            return matchOB.group()

def merge_doc_list(response_dict):
    '''
    辞書作成向け
    検索結果のtitleとsearch_field(先頭500文字)を結合し、全DOCを結合したものを返却
    '''
    #doc_list = [doc['title'] + " " + doc['search_field'][:500] for doc in response_dict['response']['docs']]
    # title + job_fieldに変更 job_fieldが空だったらsearch_fieldを利用
    #doc_list = [doc['title'] + " " + doc['job_field'] if len(doc['job_field']) != 0 else doc['title'] + " " + doc['search_field']  for doc in response_dict['response']['docs']]
    doc_list = [doc['title'] + " " + doc.get('job_field',doc['search_field'])  for doc in response_dict['response']['docs']]
    doc = ' '.join(doc_list)

    return doc


def merge_doc_dic(response_dict):
    '''
    予測データ向け
    検索結果のtitleとsearch_field(先頭500文字)を結合し、kyujin_uniqueidをkeyとしたdictを返却
    '''
    doc_dict = {}
    for doc in response_dict['response']['docs']:
        # doc_dict[doc['kyujin_uniqueid']] = doc['title'] + " " + doc['search_field'][:500]
        # title + job_fieldに変更
        doc_dict[doc['kyujin_uniqueid']] = doc['title'] + " " + doc['job_field'][:500]

    return doc_dict


def get_contents():
    '''
    検索結果をdictでまとめる
    key:label + kyujin_uniqueid
    value:title + job_field
    '''

    job_dic = get_job_dic()

    ret = {}
    for label,kwd in job_dic.items():
        # title + job_fieldを検索対象に変更
        search_kwd = "((title:" + kwd + ") OR (job_field:" + kwd + "))"
        search_kwd = urllib.parse.quote(search_kwd.encode('utf-8'))
        #url = "http://%s:%s/solr/rb_s/select?q=%s&wt=json&indent=true&fl=kyujin_uniqueid,title,job_field,search_field&sort=score%%20desc&rows=100" % (SEARCH_HOST, PORT, search_kwd)
        url = "http://%s:%s/solr/rb_s/select?q=%s&wt=json&indent=true&fl=kyujin_uniqueid,title,job_field,search_field&sort=score%%20desc&rows=500" % (SEARCH_HOST, PORT, search_kwd)
        response_dict = search_job(url)
        for doc in response_dict['response']['docs']:
            u_key = label + "_" + doc['kyujin_uniqueid']
            doc_info = doc['title'] + " " + doc.get('job_field',doc['search_field'])
            ret[u_key] = doc_info

    return ret

def get_job_dic():
    '''
    職種と検索キーワードの取得
    '''

    f = open(JOB_FILE)
    jobs = json.load(f)
    f.close()
    return jobs

def get_tokenize(contents):
    '''
    検索結果のdictを形態素解析して返却
    リストで返却
    '''
    token_list = []
    for k, content in contents.items():
        token_list.append([token for token in tokenize(content) if not check_stopwords(token)])
    return token_list

def get_tokenize2(contents):
    '''
    検索結果のdictを形態素解析して返却
    辞書形式で返却
    '''
    token_list = {}
    for k, content in contents.items():
        token_list[k] = ([token for token in tokenize(content) if not check_stopwords(token)])
        #print([token for token in tokenize(content) if not check_stopwords(token)])
    return token_list


def tokenize(text):
    '''
    とりあえず形態素解析して名詞だけ取り出す感じにしてる
    '''
    mecab = MeCab.Tagger('mecabrc')
    # neologd辞書を指定
    #mecab = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    # python3ではこれをやっておかないdecodeエラーとなる
    # https://shogo82148.github.io/blog/2015/12/20/mecab-in-python3-final/
    mecab.parse('')

    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next
    raise StopIteration()

def check_stopwords(word):
    '''
    ストップワードだったらTrueを返す
    '''
    if re.search(r'^[0-9]+$', word):  # 数字だけ
        return True
    return False

def filter_dictionary(dictionary):
    '''
    低頻度と高頻度のワードを除く感じで
    '''
    # no_below=N   N回以下の出現頻度の単語を無視 → 低頻度の単語を除外
    # no_above=0.N   Nパーセント以上に出現したワードは一般的すぎるワードとして無視
    dictionary.filter_extremes(no_below=5, no_above=0.7)  # この数字はあとで変えるかも
    return dictionary


def count_features(dictionary, content):
    '''
    ある求人の特徴語カウント
    '''
    # tokenize
    tokenized_word = [token for token in tokenize(content) if not check_stopwords(token)]
    # 辞書中の単語IDと頻度のタプル生成(タプル数=次元?) 辞書にないものは無視される
    # ベースとなる辞書がいけてないと、大事な単語が抜け落ちたり、ゴミが混じったりする
    corpus = dictionary.doc2bow(tokenized_word)

    # tf-idfかけるためここでreturn
    return corpus


def weight_tfidf(corpus):
    tfidf_model = models.TfidfModel(corpus)
    tfidf_corpus = tfidf_model[corpus]
    return tfidf_corpus

def get_dictionary(create_flg=False):
    '''
    コーパス作成
    '''
    if create_flg:

        # 各職種ごとの検索結果を取得
        # 職種名をキーとした検索結果(辞書形式)
        contents = get_contents()

        # 形態素解析
        tokenized_word = get_tokenize(contents)

        # 単語辞書作成後、filter
        #dictionary = filter_dictionary(corpora.Dictionary(tokenized_word))

        # 特徴語辞書作成
        dictionary = corpora.Dictionary(tokenized_word)

        # 辞書のファイル出力
        dictionary.save_as_text(DICT_FILE)
        
    else:
        # 通常は読み込みのみ
        dictionary = corpora.Dictionary.load_from_text(DICT_FILE)

    return dictionary

if __name__ == '__main__':
    get_dictionary(create_flg=True)
