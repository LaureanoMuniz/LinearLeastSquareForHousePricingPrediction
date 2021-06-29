import experiments.nombres as nombres


def precio_m2(df):
    df[nombres.PRECIO_M2] = df[nombres.PRECIO] / df[nombres.METROS_TOT]
    return df
