# from dataclasses import dataclass
from experiments.experiment2 import add_knearest
import experiments.nombres as nombres
from experiments.filter import filter_city
import matplotlib.pyplot as plt
import numpy as np

import build.metnum as metnum
# import sklearn.linear_model as metnum
import experiments.métricas as métricas
from experiments.kfold import kfold


from experiments.cities import ciudades


def para_ciudad(ciudad, df, tipo, con_knear=False):
    df = filter_city(df, ciudad)
    df = df[df[nombres.TIPO_DE_PROPIEDAD] == tipo]
    title = f"KNearest: {con_knear}, {ciudad.nombre}, {tipo}"
    print(title)
    errors = []
    metrics = np.zeros((0, 3))
    for train, test in kfold(df):
        if con_knear:
            train = add_knearest(train, train)
            test = add_knearest(train, test)

        features = [
            nombres.METROS_CUB,
            nombres.METROS_TOT,
            nombres.HABITACIONES,
            nombres.BAÑOS,
            nombres.GARAJES,
            nombres.GIMNASIO,
            nombres.USOS_MULT,
            nombres.PILETA,
            nombres.ESCUELAS,
            nombres.COMERCIOS,
            nombres.SEGURIDAD,
            nombres.SALAS,
            nombres.NATURALEZA,
            nombres.CHETOS,
            nombres.ONE,
        ]

        if con_knear:
            features.append(nombres.PRECIO_M2_NEAREST)

        predict = [
            nombres.PRECIO,
        ]

        regressor = metnum.LinearRegression()
        regressor.fit(train[features], train[predict])

        predicted = regressor.predict(test[features])[:, 0]
        actual = test[nombres.PRECIO].values
        # for i in range(len(predicted)):
        # print(predicted[i], test.iloc[i][nombres.PRECIO])
        # print(len(predicted))

        errors = np.append(errors, actual-predicted)
        metrics = np.append(metrics, np.array([[
            métricas.rmse(actual, predicted),
            métricas.rmsle(actual, predicted),
            métricas.r2(actual, predicted),
        ]]), axis=0)
        # print(métricas.rmsle(actual, predicted))

    plt.figure()
    print(metrics)
    print(metrics.mean(axis=0))
    plt.title(title)
    plt.hist(errors, bins=100)
    plt.savefig(f"images/{title}.png")

    # plt.figure()
    # plt.scatter(x=df[LONGITUD], y=df[LATITUD])
    # plt.show()
    # for i, row in df.iterrows():
    #     # print(row)
    #     print(row[BELLAS])
    #     for word in parser.parser(row[DESCRIPCIÓN]):
    #         if word not in words:
    #             words[word] = 0
    #         words[word] += 1
    # for word in sorted(words, key=lambda x: words[x]):
    #     print(word, words[word])


def experimento1(df, con_knear=False):
    df = df.sample(frac=0.01)
    tipos = ['Casa', 'Apartamento']
    if con_knear:
        tipos = ['Casa']
    for tipo in tipos:
        for ciudad in ciudades:
            para_ciudad(ciudad, df, tipo, con_knear=con_knear)
