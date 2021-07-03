from experiments.geo import distance
import experiments.nombres as nombres


def add_knearest(train, test):
    test = test.copy()
    # print(len(train))
    # train = train.sample(100)

    def info(row):
        suma = 0
        weight = 0
        for i, t in train.iterrows():
            # print(i)
            d = distance(
                t[nombres.LATITUD], t[nombres.LONGITUD],
                row[nombres.LATITUD], row[nombres.LONGITUD])
            w = 0 if d < nombres.eps else 1/d
            weight += w
            suma += w * t[nombres.PRECIO]
        # print("row done")
        return suma / weight
    res = test.apply(info, axis=1)
    test[nombres.PRECIO_M2_NEAREST] = res
    return test
