# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import datetime
import scipy.spatial.distance as dist
import scipy.sparse as sp


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. Odleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    X = X.toarray()
    X_train = X_train.toarray()
    return (X @ ((~X_train).transpose()).astype(np.uint8)) + ((~X).astype(np.uint8) @ X_train.transpose())


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """

    index_array = np.argsort(Dist, kind='mergesort')
    return np.fromfunction(lambda i, j: y[index_array[i, j]], (Dist.shape[0], Dist.shape[1]), dtype=int)
    # return np.array([y[index_array[int(x / N2), x % N2]] for x in range(N1 * N2)]).reshape((N1, N2))


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """

    def f(yi, kj):
        return np.count_nonzero(y[yi, :k] == (kj + 1)) / k

    f = np.vectorize(f)
    return np.fromfunction(f, shape=(y.shape[0], 4), dtype=int)


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    N1 = np.shape(p_y_x)[0]
    result = 0
    for i in range(N1):
        a = p_y_x[i].tolist()
        if (4 - a[::-1].index(max(a)) != y_true[i]):
            result += 1
    return result / N1


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbiór danych walidacyjnych N1xD
    :param Xtrain: zbiór danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    y_sorted = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    error_values = [classification_error(p_y_x_knn(y_sorted, k), yval) for k in k_values]
    min_error = min(error_values)
    return (min_error, k_values[error_values.index(min_error)], error_values)


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    Ninverse = 1 / ytrain.shape[0]
    array = np.zeros(shape=(4))
    for y in ytrain:
        array[y - 1] += Ninverse
    return array


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    # Xtrain = Xtrain.toarray()
    # a_priori = estimate_a_priori_nb(ytrain) * Xtrain.shape[0]
    # upAddition = a - 1.0
    # downAddition = a + b - 2.0
    #
    # result = np.zeros(shape=(4, Xtrain.shape[1]))
    # for d in range(Xtrain.shape[1]):
    #     for k in range(4):
    #         result[k, d] = (upAddition + sum((ytrain == k + 1) & (Xtrain[:, d] == 1)))
    #     result[:, d] /= (downAddition + a_priori)
    #
    # return result

    Xtrain = Xtrain.toarray()
    up_factor = a - 1.0
    down_factor = a + b - 2.0


    def f(k, d):
        I_yn_k = (ytrain == k + 1).astype(bool)
        I_xnd_1 = (Xtrain[:, d] == 1).astype(bool)
        up = up_factor + np.count_nonzero(I_yn_k & I_xnd_1)
        down = down_factor + np.count_nonzero(I_yn_k)
        return up / down


    g = np.vectorize(f)
    return np.fromfunction(g, shape=(4, Xtrain.shape[1]), dtype=int)

    # def f(k, d):
    #     up = upAddition + sum((ytrain == k + 1) & (Xtrain[:, d] == 1))
    #     down = downAddition + a_priori[k]
    #     # for n in range(N):
    #     #     if ((ytrain[n] == k + 1) and (Xtrain[n, d] == 1)):
    #     #         up += 1.0
    #
    #     return up / down
    #
    # g = np.vectorize(f)
    # return np.fromfunction(g, shape=(4, Xtrain.shape[1]), dtype=int)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    N = np.shape(X)[0]
    M = np.shape(p_y)[0]
    X = X.toarray()

    # temp_array = np.prod(np.array([np.array(
    #     [np.array([p_x_1_y[m, d] if (X[n, d] == 1) else (1 - p_x_1_y[m, d]) for n in range(N)]) for m in range(M)]) for
    #                                d in range(D)]), axis=0).transpose()
    # temp_2 = np.array([temp_array[n] * p_y for n in range(N)])
    # array = np.array([temp_2[n] / sum(temp_2[n]) for n in range(N)])
    # return array

    def f(n, m):
        return np.prod(np.negative(X[n, :]) - p_x_1_y[m, :])

    g = np.vectorize(f)
    result = np.fromfunction(g, shape=(N, M), dtype=int) * p_y
    result /= result @ np.ones(shape=(4, 1))

    return result


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """

    A = len(a_values)
    B = len(b_values)

    p_y = estimate_a_priori_nb(ytrain)

    def f(a, b):
        p_x_y_nb = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
        p_y_x = p_y_x_nb(p_y, p_x_y_nb, Xval)
        err = classification_error(p_y_x, yval)
        return err

    g = np.vectorize(f)
    errors = np.fromfunction(g, shape=(A, B), dtype=int)

    min = np.argmin(errors)
    minA = min // A
    minB = min % A
    return (errors[minA, minB], a_values[minA], b_values[minB], errors)
