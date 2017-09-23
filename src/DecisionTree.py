from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from PreProcess import *

vectorizer = TfidfVectorizer(max_df=0.5, max_features=5000, min_df=2, use_idf=True, lowercase=False,
                             decode_error='ignore', analyzer=str.split)


def DecisionTreeClassify(normal, spam):
    data = Data(normal, spam)
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 max_features=5000,
                                 min_df=2,
                                 use_idf=True,
                                 lowercase=False,
                                 decode_error='ignore',
                                 analyzer=str.split).fit(data.data)
    train_x, test_x, train_y, test_y = train_test_split(data.data, data.label, test_size=0.1)
    train_x = vectorizer.transform(train_x)
    test_x = vectorizer.transform(test_x)

    DecisionTree = DecisionTreeClassifier(criterion="entropy",
                                          splitter="best",
                                          max_depth=None,
                                          min_samples_split=2,
                                          min_samples_leaf=2)
    DecisionTree.fit(train_x, train_y)
    print(DecisionTree.score(test_x, test_y))

    words = vectorizer.get_feature_names()
    feature_importance = DecisionTree.feature_importances_
    word_importances_dict = dict(zip(words, feature_importance))
    word_importances_dict = dict(sorted(word_importances_dict.items(), key=lambda val:val[1], reverse=True))

    number = 0
    for item in word_importances_dict:
        number += 1
        print(item, end=" ")
        if number == 200:
            break

DecisionTreeClassify('../data/normal.txt', '../data/spam.txt')
