from experiments.nombres import LONGITUD, LATITUD, METROS_TOT, METROS_CUB,\
    HABITACIONES, BAÑOS, GARAJES, TÍTULO, GIMNASIO, USOS_MULT, PILETA, \
    ESCUELAS, COMERCIOS, PRECIO, DESCRIPCIÓN, eps


def filter(df):
    useful = df[LONGITUD].notnull() & df[LONGITUD].abs() > eps
    useful &= df[LATITUD].notnull() & df[LATITUD].abs() > eps
    useful &= df[METROS_TOT].notnull() & df[METROS_TOT].abs() > eps
    useful &= df[METROS_CUB].notnull() & df[METROS_CUB].abs() > eps
    useful &= df[HABITACIONES].notnull() & df[HABITACIONES].abs() > eps
    useful &= df[PRECIO].notnull() & df[PRECIO].abs() > eps
    useful &= df[BAÑOS].notnull()
    useful &= df[GARAJES].notnull()
    useful &= df[TÍTULO].notnull()
    useful &= df[GIMNASIO].notnull()
    useful &= df[USOS_MULT].notnull()
    useful &= df[PILETA].notnull()
    useful &= df[ESCUELAS].notnull()
    useful &= df[COMERCIOS].notnull()
    useful &= df[DESCRIPCIÓN].notnull()
    return df[useful]


def filter_city(df, ciudad):
    in_city = (df[LATITUD] > ciudad.lat_lo)\
        & (df[LATITUD] < ciudad.lat_hi)\
        & (df[LONGITUD] > ciudad.lng_lo)\
        & (df[LONGITUD] < ciudad.lng_hi)

    return df[in_city]
