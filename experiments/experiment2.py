from experiments.geo import distance
import experiments.nombres as nombres


def add_knearest(train, test):
    def info(row):
        suma = 0
        weight = 0
        for t in train.iterrows():
            d = distance(
                t[nombres.LATITUD], t[nombres.LONGITUD],
                row[nombres.LATITUD], row[nombres.LONGITUD])
            w = 0 if d < nombres.eps else 1/d
            weight += w
            suma += w * t[nombres.PRECIO_M2]
        row[nombres.PRECIO_M2_NEAREST] = suma / weight
    return test.apply(info, axis=1)
