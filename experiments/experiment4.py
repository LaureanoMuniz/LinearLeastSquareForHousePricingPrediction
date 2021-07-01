import experiments.nombres as nombres
import build.metnum as metnum
import math


def folded(train, test):
    features = [
        nombres.METROS_CUB,
    ]

    predict = [
        nombres.HABITACIONES,
    ]

    regressor = metnum.LinearRegression()
    regressor.fit(train[features], train[predict])

    predicted = regressor.predict(test[features])
    error = math.sqrt(((predicted - test[predict])**2).sum() / len(predicted))

    print(error)


def experiment4(df):
    df = df[df[nombres.TIPO_DE_PROPIEDAD] == "Casa"]
    n = len(df)
    folded(df[:n*8//10], df[n*8//10:])
