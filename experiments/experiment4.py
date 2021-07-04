from experiments.kfold import kfold
import experiments.nombres as nombres
import build.metnum as metnum
# import sklearn.linear_model as metnum
# import math
import experiments.métricas as métricas
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


features = [
    nombres.METROS_CUB,
    nombres.ONE,
]

predict = [
    nombres.HABITACIONES,
]


def folded(train, test):
    regressor = metnum.LinearRegression()
    # print("Shape A", train[features].values.shape)
    # print("Shape B", train[predict].values.shape)
    regressor.fit(train[features].values, train[predict].values)

    # coeffs = regressor._coeffs()
    # print(coeffs.shape)
    # print(coeffs)
    # predicted = test[features].values @ coeffs
    predicted = regressor.predict(test[features].values)
    actual = test[predict].values

    rmse = métricas.rmse(actual, predicted)
    rmsle = métricas.rmsle(actual, predicted)
    r2 = métricas.r2(actual, predicted)

    return np.array([rmse, rmsle, r2])


def iterate(df):
    for train, test in kfold(df):
        yield folded(train, test)


def gráfico(df):
    regressor = metnum.LinearRegression()
    regressor.fit(df[features], df[predict])

    df[predict[0]] += (np.random.binomial(1000, 0.5, size=len(df)) - 500) * 0.005
    df['predicted'] = regressor.predict(df[features])

    sns.set()
    sns.scatterplot(x=features[0], y=predict[0], data=df, s=3)
    sns.lineplot(x=features[0], y='predicted', data=df)

    plt.show()


def experiment4(df):
    tipos = ["Casa", "Apartamento"]
    for tipo in tipos:
        este = df[df[nombres.TIPO_DE_PROPIEDAD] == tipo]

        errors = np.array(list(iterate(este)))
        print(tipo)
        print(errors.mean(axis=0))
        gráfico(este)
