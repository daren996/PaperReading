import random
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


class Vectorize:

    def __init__(self, data_raw):
        """
        Initialization
        :param data_raw: [doc1, doc2, ...]
        """
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(data_raw)

    def vectorize(self, data_set_raw):
        """
        To vectorize raw data.
        :param data_set_raw: [doc1, doc2, ...]
        :return:
        data_vec_: vectorized data [numbers]
        """
        return self.vectorizer.transform(data_set_raw)

    def get_dictionary(self):
        return self.vectorizer.get_feature_names()


class DecisionTree:

    def __init__(self, max_depth, criterion):
        self.clf = tree.DecisionTreeClassifier(criterion=criterion, splitter='best', max_depth=max_depth,
                                               min_samples_split=2, min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, random_state=None)

    def fit_train_data(self, train_data_vec_, train_target_):
        self.clf.fit(train_data_vec_.toarray(), train_target_)

    def get_predict(self, test_data_vec_):
        return self.clf.predict(test_data_vec_)

    def get_score(self, test_data_vec_, test_target_):
        return self.clf.score(test_data_vec_.toarray(), test_target_)

    def get_feature_importance(self):
        print(self.clf.feature_importances_)
        return self.clf.feature_importances_

    def get_cross_val_score(self, train_data_vec_, train_target_, cv):
        return cross_val_score(self.clf, train_data_vec_, train_target_, cv=cv)

    def save_tree_graph(self, out_file_name, feature_names):
        with open(out_file_name, "w") as out_file:
            tree.export_graphviz(self.clf, feature_names=feature_names, filled=True,
                                 class_names=['fake', 'real'], out_file=out_file)


def split_dataset(data_set_, divide_point):
    """
    Shuffle the dataset and split it into train, validation and test dataset.
    :param data_set_: [(doc, target), (...), ...]
    :param divide_point: points for division
    :return: split_set: [set1, set2, set3, ...], set1 = ([doc1, doc2, ...], [1, 1, 0, ...])
    """
    random.shuffle(data_set_)
    # train_set_ = data_set_[:int(0.7 * len(data_set_))]
    # valid_set_ = data_set_[int(0.7 * len(data_set_)):int(0.85 * len(data_set_))]
    # test_set_ = data_set_[int(0.85 * len(data_set_)):]
    split_set = []
    data_set_x, data_set_y = [ds[0] for ds in data_set_], [ds[1] for ds in data_set_]
    for i in range(len(divide_point)):
        temp_train_x, temp_test_x, temp_train_y, temp_test_y = \
            train_test_split(data_set_x, data_set_y, test_size=1-divide_point[i])
        split_set.append((temp_train_x, temp_train_y))
        data_set_x, data_set_y = temp_test_x, temp_test_y
    split_set.append((data_set_x, data_set_y))
    return split_set


def load_data(fake_file, real_file, divide_point):
    """
    Load, split and vectorize data from fake and real news files.
    :param fake_file: fake news file
    :param real_file: real news file
    :param divide_point: points for division (should take into account that once divide it a smaller total left)
    :return: train_vec, valid_vec，test_vec: [vec_docs]; to use them, train_vec.toarray()
    :return: train_target, valid_target, test_target：[targets]
    :return: dictionary: [features]
    """
    data_set_ = []
    with open(fake_file, "r") as in_file:
        for line in in_file.readlines():
            data_set_.append((line.strip(), 0))
    with open(real_file, "r") as in_file:
        for line in in_file.readlines():
            data_set_.append((line.strip(), 1))
    # split
    split_set = split_dataset(data_set_, divide_point)
    train_raw, valid_raw, test_raw = split_set[0], split_set[1], split_set[2]  # train_raw = ([docs], [targets])
    print("Size of divided dataset: train:%d, valid:%d test:%d" %
          (len(train_raw[0]), len(valid_raw[0]), len(test_raw[0])))
    # vectorize
    vectorizer_ = Vectorize([ds[0] for ds in data_set_])
    train_vec_ = vectorizer_.vectorize(train_raw[0])
    valid_vec_ = vectorizer_.vectorize(valid_raw[0])
    test_vec_ = vectorizer_.vectorize(test_raw[0])
    return train_vec_, train_raw[1], valid_vec_, valid_raw[1], test_vec_, test_raw[1], vectorizer_


def select_model(train_vec_, train_target_, valid_vec_, valid_target_, depth_set_):
    """
    Train the classifier using different values of max_depth,
    as well as two different split criteria (information gain
    and Gini coefficient, "entropy" and "gini").
    :param: train_vec_, train_target_, valid_vec_, valid_target_
    :param: depth_set_: list of different values of max_depth, [values]
    :return: decision_tree_best_, best_criterion_, best_depth_
    """
    best_acc_, best_criterion_, best_depth_ = 0, "", None
    decision_tree_best_ = None
    for criterion_ in ["gini", "entropy"]:
        for max_depth_ in list(depth_set_):
            decision_tree_ = DecisionTree(max_depth=max_depth_, criterion=criterion_)
            decision_tree_.fit_train_data(train_vec_, train_target_)
            acc_ = decision_tree_.get_score(valid_vec_, valid_target_)
            print("%s & %d & %.4f" % (criterion_, max_depth_, acc_))
            if acc_ > best_acc_:
                best_acc_ = acc_
                decision_tree_best_ = decision_tree_
                best_criterion_ = criterion_
                best_depth_ = max_depth_
    print("Best decision tree found, accuracy: %.5f, best criterion: %s, best depth: %d" %
          (best_acc_, best_criterion_, best_depth_))
    return decision_tree_best_, best_criterion_, best_depth_


if __name__ == '__main__':
    train_vec, train_target, valid_vec, valid_target, test_vec, test_target, vectorizer = \
        load_data("./fake_news.txt", "./real_news.txt", divide_point=[0.7, 0.5])

    depth_set = [5, 10, 12, 14, 15, 16, 18, 20, 25, 30, 35, 40]
    decision_tree, best_criterion, best_depth = \
        select_model(train_vec, train_target, valid_vec,
                     valid_target_=valid_target, depth_set_=depth_set)

    decision_tree.save_tree_graph("./tree.dot", vectorizer.get_dictionary())

    print("test set acc:", decision_tree.get_score(test_vec, test_target))
