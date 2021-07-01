from experiments.kfold import split
import experiments.nombres as nombres
import experiments.métricas as métricas
# import build.metnum as metnum
import sklearn.linear_model as metnum
from experiments.filter import filter_city
from experiments.cities import ciudades

import matplotlib.pyplot as plt


def dado_subconjunto(features, df):
    train, test = split(df)
    regressor = metnum.LinearRegression()
    regressor.fit(train[features], train[nombres.PRECIO])

    predicted = regressor.predict(test[features])
    return predicted


def dado_subconjunto_cp(features, df, total):
    train, test = split(df)
    regressor = metnum.LinearRegression()
    regressor.fit(train[features], train[nombres.PRECIO])

    predicted = regressor.predict(test[features])
    # print(predicted.shape)
    # print(total.shape)
    actual = test[nombres.PRECIO].values
    return métricas.cp(actual, predicted, len(features), total)


def powerset(elements, i=0):
    if i == len(elements):
        yield []
    else:
        for v in powerset(elements, i+1):
            yield v
            v = v[:]
            v.append(elements[i])
            yield v


def todos_los_subconjuntos(df):
    features = [
        nombres.METROS_CUB,
        nombres.METROS_TOT,
        nombres.HABITACIONES,
        nombres.BAÑOS,
        nombres.GARAJES,
        nombres.GIMNASIO,
        nombres.USOS_MULT,
        nombres.PILETA,
    ]
    total = dado_subconjunto(features, df)
    result = dict()
    for s in powerset(features):
        if len(s) != 0:
            # print(s)
            result[tuple(s)] = dado_subconjunto_cp(s, df, total)
    return result


def experiment5(df):
    df = filter_city(df, ciudades[0])
    result = todos_los_subconjuntos(df)

    xs = []
    ys = []
    for r in result:
        # print(r, result[r])
        xs.append(len(r))
        ys.append(result[r])

    plt.figure()
    plt.scatter(x=xs, y=ys)
    plt.show()
