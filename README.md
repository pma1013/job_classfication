# job_classfication

## 前提条件
python3かつ下記のライブラリがインストール済みであること
* MeCab
* mecab-python3
* scikit-learn
* gensim

## 特徴語辞書の作成
```
$ python corpus.py
```

## 職種分類
```
$ python estimation.py
==== 学習データと予測データが一緒の場合
0.994615384615
==== 学習データと予測データが違う場合
0.9
```
