# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: k-NN and Naive Bayes
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
from scipy import spatial
import numpy as np


def hamming_distance(X, X_train):
    """
    :param X: set of objects that are going to be compared N1xD
    :param X_train: set of objects compared against param X N2xD
    Functions calculates Hamming distances between all objects from set X  and all object from set X_train.
    Resulting distances are returned as matrices.
    :return: Distance matrix between objects X and X_train X i X_train N1xN2
    """
    X = X.toarray()
    X_train = X_train.toarray()
    # ans = np.zeros((X.shape[0], X_train.shape[0]))
    # i = 0
    # for e in X:
    #     l = [None] * len(X_train)
    #     j = 0
    #     for f in X_train:
    #         l[j] = spatial.distance.hamming(e, f) * len(f)
    #         # l.append(spatial.distance.hamming(e, f)*len(f))
    #         # l.append(sum(e ^ f))
    #         j += 1
    #     ans[i] = l
    #     i += 1
    # return ans
    return spatial.distance.cdist(X, X_train, metric='hamming')*X.shape[1]


def sort_train_labels_knn(Dist, y):
    """
    Function sorts labels of training data y accordingly to probabilities stored in matrix Dist.
    Function returns matrix N1xN2. In each row there are sorted data labels y accordingly to corresponding row of matrix Dist.
    :param Dist: Distance matrix between objects X and X_train N1xN2
    :param y: N2-element vector of labels
    :return: Matrix of sorted class labels ( use mergesort algorithm)
    """
    index_array = np.argsort(Dist, kind='mergesort')
    return np.fromfunction(lambda i, j: y[index_array[i, j]], (Dist.shape[0], Dist.shape[1]), dtype=int)


def p_y_x_knn(y, k):
    """
    Function calculates conditional probability p(y|x) for
    all classes and all objects from test set using KNN classifier
    :param y: matrix of sorted labels for training set N1xN2
    :param k: number of nearest neighbours
    :return: matrix of probabilities for objects X
    """
    topics = y.max()
    ans = np.zeros((y.shape[0], topics))
    i = 0
    for x in y:
        for j in range(0, topics):
            ans[i][j] = (x[:k] == j + 1).sum() / k
        i += 1
    return ans


def classification_error(p_y_x, y_true):
    """
    Function calculates classification error
    :param p_y_x: matrix of predicted probabilities
    :param y_true: set of ground truth labels 1xN.
    Each row of matrix represents distribution p(y|x)
    :return: classification error
    """
    error = 0
    for i in range(0, len(p_y_x)):
        if y_true[i] != max(np.where(max(p_y_x[i]) == p_y_x[i])[0]) + 1:
            error += 1
    return error / len(p_y_x)


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: validation data N1xD
    :param Xtrain: training data N2xD
    :param yval: class labels for validation data 1xN1
    :param ytrain: class labels for training data 1xN2
    :param k_values: values of parameter k that are going to be evaluated
    :return: function makes model selection with knn and results tuple best_error,best_k,errors), where best_error is the lowest
    error, best_k - value of k parameter that corresponds to the lowest error, errors - list of error values for
    subsequent values of k (elements of k_values)
    """
    ySorted = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    errors = [None] * len(k_values)
    for i in range(0, len(k_values)):
        errors[i] = classification_error(p_y_x_knn(ySorted, k_values[i]), yval)
    minError = min(errors)
    return minError, k_values[errors.index(minError)], errors


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: labels for training data 1xN
    :return: function calculates distribution a priori p(y) and returns p_y - vector of a priori probabilities 1xM
    """
    k = max(ytrain)
    ans = [None] * k
    for i in range(0, k):
        ans[i] = (ytrain == i + 1).sum() / len(ytrain)
    return ans


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: training data NxD
    :param ytrain: class labels for training data 1xN
    :param a: parameter a of Beta distribution
    :param b: parameter b of Beta distribution
    :return: Function calculated probality p(x|y) assuming that x takes binary values and elements
    x are independent from each other. Function returns matrix p_x_y that has size MxD.
    """
    # k = max(ytrain)
    # Xtrain = Xtrain.toarray()
    # a = a[0]
    # b = b[0]
    # D = Xtrain.shape[1]
    # theta = np.zeros((k, D))
    # for i in range(0, k):
    #     for j in range(0, D):
    #         licznik = sum([ytrain[n] == i + 1 and Xtrain[n][j] == 1 for n in range(0, len(Xtrain))]) + a - 1
    #         mianownik = (ytrain == i + 1).sum() + a + b - 2
    #         theta[i][j] = licznik / mianownik
    # return theta
    k = max(ytrain)
    Xtrain = Xtrain.toarray()
    a = a[0]
    b = b[0]
    D = Xtrain.shape[1]
    theta = np.zeros((k, D))
    for i in range(0, k):
        for j in range(0, D):
            licznik = sum((ytrain == i + 1) & (Xtrain[:,j] == 1)) + a - 1
            mianownik = (ytrain == i + 1).sum() + a + b - 2
            theta[i][j] = licznik / mianownik
    return theta

def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: vector of a priori probabilities 1xM
    :param p_x_1_y: probability distribution p(x=1|y) - matrix MxD
    :param X: data for probability estimation, matrix NxD
    :return: function calculated probability distribution p(y|x) for each class with the use of Naive Bayes classifier.
     Function returns matrixx p_y_x of size NxM.
    """
    X = X.toarray()
    D = X.shape[1]
    N = X.shape[0]
    M = p_y.shape[0]
    ans = np.zeros((N, M))
    return ans


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: training setN2xD
    :param Xval: validation setN1xD
    :param ytrain: class labels for training data 1xN2
    :param yval: class labels for validation data 1xN1
    :param a_values: list of parameters a (Beta distribution)
    :param b_values: list of parameters b (Beta distribution)
    :return: Function makes a model selection for Naive Bayes - that is selects the best values of a i b parameters.
    Function returns tuple (error_best, best_a, best_b, errors) where best_error is the lowest error,
    best_a - a parameter that corresponds to the lowest error, best_b - b parameter that corresponds to the lowest error,
    errors - matrix of errors for each pair (a,b)
    """
    pass
