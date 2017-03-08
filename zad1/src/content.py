# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from utils import polynomial
from numpy.linalg import inv


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    # print("1: ", x)
    # print("2: ", y)
    # print("3: ", w)
    # print("4: ", polynomial(x, w))
    # print("5: ", y - polynomial(x, w))
    # print("5: ", (y - polynomial(x, w)) ** 2)
    # print("6: ",((y - polynomial(x, w)) ** 2).sum())
    # print("7: ",x.shape[0])
    # print("8: ",((y - polynomial(x, w)) ** 2).sum() / x.shape[0])
    # np.set_printoptions(precision=2, suppress=True)
    return ((y - polynomial(x, w)) ** 2).sum() / y.shape[0]


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''

    m = x_train @ np.ones(shape=(1, M + 1))
    for i in range(M + 1):
        m[:, i] **= i
    return m


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''

    phi = design_matrix(x_train, M)
    w = inv(phi.transpose() @ phi) @ phi.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''

    phi = design_matrix(x_train, M)
    identity = np.identity(phi.shape[1])
    w = inv(phi.transpose() @ phi + regularization_lambda * identity) @ phi.transpose() @ y_train
    return w, mean_squared_error(x_train, y_train, w)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''

    train_array = []
    for m in M_values:
        w, train_err = least_squares(x_train, y_train, m)
        train_array.append((w, train_err, mean_squared_error(x_val, y_val, w)))
    return min(train_array, key=lambda t: t[2])


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''

    train_array = []
    for regularization_lambda in lambda_values:
        w, train_err = regularized_least_squares(x_train, y_train, M, regularization_lambda)
        train_array.append((w, train_err, mean_squared_error(x_val, y_val, w),regularization_lambda))
    return min(train_array, key=lambda t: t[2])
