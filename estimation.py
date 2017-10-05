# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
import corpus
from gensim import corpora, models, matutils

def learning_TruncatedSVD():
    # 辞書の読み込み
    dictionary = corpus.get_dictionary(create_flg=False)
    # 記事の読み込み
    contents = corpus.get_contents()

    # 特徴抽出
    data_train = []
    label_train = []
    for u_key, content in contents.items():
        data_train.append(corpus.get_vector(dictionary, content))
        label_train.append(corpus.get_label(u_key))

    # tf-idf
    tfidf_corpus = corpus.weight_tfidf(data_train)

    # LSIモデル構築
    #lsi_instance=models.LsiModel(corpus=tfidf_corpus, num_topics=300, id2word=dictionary);
    #lsi_instance=models.LsiModel(corpus=tfidf_corpus, num_topics=50, id2word=dictionary);
    #lsi_corpus=lsi_instance[tfidf_corpus];

    # ベクトル変換
    #dense = matutils.corpus2dense(lsi_corpus, num_terms=len(dictionary))
    #data_train = [ matutils.corpus2dense([i], num_terms=len(dictionary)).T[0] for i in lsi_corpus ]
    data_train = [ matutils.corpus2dense([i], num_terms=len(dictionary)).T[0] for i in tfidf_corpus ]

    #import pdb; pdb.set_trace()

    # 次元圧縮
    #svd = TruncatedSVD(300) # 500次元まで削減
    #svd_corpus = svd.fit(lsi_corpus)
    #svd_corpus = svd.fit(tfidf_corpus)

    # 分類器
    estimator = RandomForestClassifier()

    # 学習
    estimator.fit(data_train, label_train)

    # 学習したデータを予測にかけてみる（ズルなので正答率高くないとおかしい）
    print("==== 学習データと予測データが一緒の場合")
    print(estimator.score(data_train, label_train))

    # 学習データと試験データに分けてみる
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.1)

    # 分類器をもう一度定義
    estimator2 = RandomForestClassifier()

    # 学習
    estimator2.fit(data_train_s, label_train_s)
    print("==== 学習データと予測データが違う場合")
    print(estimator2.score(data_test_s, label_test_s))

def main():
    # 辞書の読み込み
    dictionary = corpus.get_dictionary(create_flg=False)
    # 記事の読み込み
    contents = corpus.get_contents()

    # 特徴カウント
    data_train = []
    label_train = []
    for u_key, content in contents.items():
        data_train.append(corpus.count_features(dictionary, content))
        label_train.append(corpus.get_label(u_key))

    # tf-idf
    tfidf_corpus = corpus.weight_tfidf(data_train)

    # ベクトル変換
    data_train = [ matutils.corpus2dense([i], num_terms=len(dictionary)).T[0] for i in tfidf_corpus ]

    # 分類器
    estimator = RandomForestClassifier()

    # 学習
    estimator.fit(data_train, label_train)

    # 学習したデータを予測にかけてみる（ズルなので正答率高くないとおかしい）
    print("==== 学習データと予測データが一緒の場合")
    print(estimator.score(data_train, label_train))

    # 学習データと試験データに分けてみる
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.1)

    # 分類器をもう一度定義
    estimator2 = RandomForestClassifier()

    # 学習
    estimator2.fit(data_train_s, label_train_s)
    print("==== 学習データと予測データが違う場合")
    print(estimator2.score(data_test_s, label_test_s))

    # グリッドサーチやってみる
    #tuned_parameters = [{'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150], 'max_features': ['auto', 'sqrt', 'log2', None]}]

    #clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=2, scoring='accuracy', n_jobs=-1)
    #clf.fit(data_train_s, label_train_s)

    #print("==== グリッドサーチ")
    #print("  ベストパラメタ")
    #print(clf.best_estimator_)

    #print("トレーニングデータでCVした時の平均スコア")
    #for params, mean_score, all_scores in clf.grid_scores_:
    #        print("{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params))

    #y_true, y_pred = label_test_s, clf.predict(data_test_s)
    #print(classification_report(y_true, y_pred))

if __name__ == '__main__':
    main()
    #learning_TruncatedSVD()

