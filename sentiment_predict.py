# -*- coding: utf-8 -*-

import os
from io import open
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, precision_score, recall_score
from SentimentIntensityAnalyzer import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.svm import SVC
from datetime import datetime
import pandas as pd
from dateutil.parser import parse as parse_datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def map_to_label(scores):
    labels = []
    for score in scores:
        if score > -0.5 and score < 0.5:
            labels.append(1)
        elif score >= 0.5:
            labels.append(2)
        elif score <= -0.5:
            labels.append(0)
    return labels

def load_tweets_processed(raw=False):
    filename = 'tweets_processed.txt'
    if raw:
        filename = 'raw_tweets_processed.txt'
    with open(os.path.join(data_dir, filename), 'r', encoding="utf-8") as f:
        tweets_processed = np.array(f.readlines())
        return tweets_processed

def load_ground_truth():
    p, neg, n = 0, 0, 0
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        y = np.array([ int(line.strip()) for line in f.readlines()])
        for i in y:
            if i == 0: p = p+1
            elif i == 1: n = n+1
            else : neg = neg+1
        print("positive count %d \nnegative count %d\nneutral count %d \n"%(p,neg,n))
        return y

def get_social_feature_mat(df):
    feature_names = ["favorite_count","retweet_count","friends_count","followers_count"]
    social_feature_mat = np.zeros([4,len(df)])
    for i in range(4):
        social_feature_mat[i] = df[feature_names[i]]
    return social_feature_mat.T

def get_temporal_feature_mat(create_at_arr):
    temporal_feats = np.zeros((len(create_at_arr), 4))
    for i, create_at in enumerate(create_at_arr):
        temporal_feats[i] = parse_time(create_at)
    return temporal_feats

def map_to_score(predict_prob):
    def get_score(vec):
        if np.argmax(vec) == 0:
            return max(vec)+0.5
        elif np.argmax(vec) == 1:
            return max(vec)-0.5
        else:
            return -max(vec)-0.5
    scores = [get_score(p_vec) for p_vec in predict_prob]
    return scores

def parse_time(created_at):
    # dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
    dt = parse_datetime(created_at)
    return [dt.hour, dt.weekday(), dt.month, dt.year-2000]

def ml_soft_vote(proba1, proba2, proba3, w1=0.2, w2=0.2, w3=0.6, averaging=False):
    weighted_proba_mat = w1*proba1 + w2*proba2 + w3*proba3
    if averaging:
        weighted_proba_mat= (proba1+  proba2 + proba3 ) / 3
    predicts = [np.argmax(vec) for vec in weighted_proba_mat]
    return predicts

def ml_hard_vote(p1, p2):
    predicts = np.zeros(len(p1))
    for i in range(len(p1)):
        majority = get_majority([p1[i], p2[i]])
        predicts[i] = majority
    return predicts

def polarity_soft_vote(s0, s1, s2, s3, s4, w=[0.9, 0.05, 0.05, 0]):
    predict_score = w[0]*s0 + w[1]*s2 + w[2]*s2 + w[3]*s3 + w[4]*s4
    predicts = map_to_label(predict_score)
    return predicts

def rule_based_hard_vote(s1, s2, s3, s4):
    predicts = np.zeros(len(s1))
    p1, p2, p3, p4 = map_to_label(s1), map_to_label(s2), map_to_label(s3), map_to_label(s4)
    for i in range(len(p1)):
        majority = get_majority([p1[i], p2[i], p3[i], p4[i]])
        predicts[i] = majority
    return predicts


def hard_vote(pred_list):
    data_size = len(pred_list[0])
    predicts = np.zeros(data_size)
    for i in range(data_size):
        preds = [p[i] for p in pred_list]
        majority = get_majority(preds)
        predicts[i] = majority
    return predicts

def get_majority(l):
    v, count = Counter(l).most_common(1)[0]
    return v

def get_polarity_scores(feats, test):
    scores = np.zeros(len(test))
    for i, instance in enumerate(feats[test[0]:test[-1]]):
        scores[i] = SentimentIntensityClassifier.polarity_scores(instance)["compound"]
    return scores

def rule_based_classify(train, test, hard_vote=False, averaging=False):
    polarity_s = get_polarity_scores(tweets_processed, test)
    hashtags_s = get_polarity_scores(hashtags, test)
    emoticon_s = get_polarity_scores(emoticons, test)
    context_s = get_polarity_scores(tweets_context, test)


    if averaging:
        return map_to_label((context_s+polarity_s+hashtags_s+emoticon_s)/4)
    elif hard_vote:
        return rule_based_hard_vote(context_s, polarity_s, hashtags_s, emoticon_s)
    else:
        return polarity_soft_vote(context_s, polarity_s, hashtags, emoticon_s)


def pla(X, Y):
    # initialise
    w = np.array([0.6, 0.3, 0.1])
    eta = 1
    epochs = 20

    for t in range(epochs):
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                w = w + eta*X[i]*Y[i]
    return w

def ml_based_classify(train, test, use_temporal=False, use_social=False, use_tweet=False, early_fusion=True):
    if use_temporal:
        clf = SVC()
        clf.fit(temporal_feats[train], y[train])
        return clf.predict(temporal_feats[test])
    elif use_social:
        clf = SVC()
        clf.fit(social_feats[train], y[train])
        return clf.predict(social_feats[test])
    elif use_tweet:
        # clf = SVC()
        # clf = MultinomialNB()
        clf = LogisticRegression()
        clf.fit(tweets_countvec[train], y[train])
        return clf.predict(tweets_countvec[test])
    elif early_fusion:
        clf = LogisticRegression()
        x_train = np.hstack((tweets_countvec[train], social_feats[train], temporal_feats[train]))
        x_test = np.hstack((tweets_countvec[test], social_feats[test], temporal_feats[test]))
        clf.fit(x_train, y[train])
        return clf.predict(x_test)
    else:
        clf1 = SVC(probability=True)
        clf1.fit(temporal_feats[train], y[train])
        clf1_proba = clf1.predict_proba(temporal_feats[test])

        clf2 = SVC(probability=True)
        clf2.fit(social_feats[train], y[train])
        clf2_proba = clf2.predict_proba(social_feats[test])

        clf3 = LogisticRegression()
        clf3.fit(tweets_countvec[train], y[train])
        clf3_proba = clf3.predict_proba(tweets_countvec[test])

        return ml_soft_vote(clf1_proba, clf2_proba, clf3_proba, averaging=True)



def vote_classify(train, test):
    tweets_polarity_p = map_to_label(get_polarity_scores(tweets_processed, test))
    hashtags_p = map_to_label(get_polarity_scores(hashtags, test))
    emoticon_p = map_to_label(get_polarity_scores(emoticons, test))
    social_p = ml_based_classify(train, test, use_social=True)
    temporal_p = ml_based_classify(train, test, use_temporal=True)
    tweets_p = ml_based_classify(train, test, use_tweet=True)
    return hard_vote([tweets_polarity_p, hashtags_p, emoticon_p, social_p, temporal_p, tweets_p])

def word2countvec(tweets_processed):
    vectorizer = CountVectorizer()
    tweets_feat = vectorizer.fit_transform(tweets_processed)
    return tweets_feat.toarray()

def word2tfidfweight(tweets_processed):
    vectorizer = TfidfVectorizer()
    tweets_feat = vectorizer.fit_transform(tweets_processed)
    return tweets_feat.toarray()

def nn_based_classify(y, model="lstm"):
    from bidirectional_lstm import bidirectional_lstm_predict
    from fasttext import fasttext_predict
    from lstm import lstm_predict
    X = word2countvec(load_tweets_processed(raw=True))
    if model == "b_lstm":
        bidirectional_lstm_predict(X, y)
    elif model == "fasttext":
        fasttext_predict(X, y)
    else:  # lstm
        lstm_predict(X, y)


def classify(train, test):
    predicts = ml_based_classify(train, test, use_tweet=True)
    return predicts

if __name__ == '__main__':

    SentimentIntensityClassifier = SentimentIntensityAnalyzer()

    data_dir = './data'
    test_data = './data/test'
    use_nn = False
    print("Loading data...")

    # load features
    tweets_processed = load_tweets_processed()
    tweets_countvec = word2countvec(tweets_processed)
    tweets_tfidf = word2tfidfweight(tweets_processed)
    df = pd.read_csv(os.path.join(data_dir,"new.csv"))
    emoticons = [str(v) for v in df["emoji_text"]]
    hashtags = [str(v) for v in df["hashtags"]]
    contexts = [str(v) for v in df["context"]]
    tweets_context = [tweets_processed[i]+
                      unicode(contexts[i], 'utf-8') for i in range(len(tweets_processed))]
    social_feats = get_social_feature_mat(df)
    temporal_feats = get_temporal_feature_mat(df["created_at"])

    y = load_ground_truth()


    if use_nn:
        nn_based_classify(y, model="b_lstm")
        exit()



    print("Start training and predict...")
    kf = KFold(n_splits=10)
    avg_p = 0
    avg_r = 0

    for train, test in kf.split(y):
        predicts = rule_based_classify(train, test, hard_vote=False, averaging=True)
        # predicts = ml_based_classify(train, test, use_tweet=False, early_fusion=False)
        # predicts = vote_classify(train, test)

        # best model
        # predicts = classify(train, test)


        print(classification_report(y[test],predicts))
        avg_p   += precision_score(y[test],predicts, average='macro')
        avg_r   += recall_score(y[test],predicts, average='macro')

    print('Average Precision is %f.' %(avg_p/10.0))
    print('Average Recall is %f.' %(avg_r/10.0))

