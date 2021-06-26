import metnum
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# plt.use('TkAgg')

from sklearn.linear_model import LinearRegression

df = pd.read_csv('../data/train.csv')

# print(df)

x = df['metroscubiertos'].values
y = df['precio'].values

linear_regressor = LinearRegression()
#linear_regressor = metnum.LinearRegression()

# linear_regressor.fit(x, y)

conlatitud = df[df['lat'].notnull()]
conlatitud = conlatitud[conlatitud['lat'] != 0.0]
conlatitud.info()
ciudades = conlatitud['ciudad'].value_counts(sort=True)[:10].index.tolist()
print(type(ciudades))
print(ciudades)


from math import cos, asin, sqrt, pi, radians


def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2

    if a < 0:
        a = 0
    if sqrt(a) > 1 or sqrt(a) < 0:
        print(a)

    a = min(1, sqrt(a))
    a = max(0, a)
    return 12742 * asin(a)


def distancias_ciudades(conlatitud):
    maximas_distancias_por_ciudad = []
    latitudes = []
    for ciudad in ciudades:
        print(ciudad)
        tmp = conlatitud[conlatitud['ciudad'] == ciudad].dropna()
        tmp = tmp[tmp['lat'] != 0.0]  # Saco los que no sirven
        print(tmp['lat'].max())
        latitudes_tmp = tmp['lat'].tolist()
        print(max(latitudes_tmp))
        if latitudes == latitudes_tmp:
            print("Son iguales")
        latitudes = latitudes_tmp
        longitudes_tmp = tmp['lng'].tolist()
        maximadist = 0
        tmp_i = 0
        tmp_j = 0
        for i in range(len(latitudes_tmp)):
            for j in range(i):
                actual = distance(latitudes_tmp[i],longitudes_tmp[i],latitudes_tmp[j],longitudes_tmp[j])
                if maximadist < actual:
                    maximadist = actual
                    tmp_i = i
                    tmp_j = j

        maximas_distancias_por_ciudad.append(maximadist)
        print(tmp_i, tmp_j)
        print(latitudes_tmp[tmp_i],longitudes_tmp[tmp_i],latitudes_tmp[tmp_j],longitudes_tmp[tmp_j])

    print(maximas_distancias_por_ciudad)
    maximas_distancias_por_ciudad.sort()
    print(maximas_distancias_por_ciudad)


def latlong():
    for ciudad in ['Coyoacán']:
        print(ciudad)
        tmp = conlatitud
        # tmp = conlatitud[conlatitud['ciudad'] == ciudad].dropna()
        tmp = tmp[tmp['tipodepropiedad'] == 'Casa'].dropna()
        # tmp = conlatitud[conlatitud['tipodepropiedad'] == 'Apartamento'].dropna()
        tmp = tmp[tmp['lat'] != 0.0]  # Saco los que no sirven

        # latmin = -100.1422
        # latmax = -98.2361

        # lngmin = 18.8145
        # lngmax = 19.8181

        latmin = -99.333
        latmax = -98.806

        lngmin = 19.252
        lngmax = 19.544

        # latmin = -99.25
        # latmax = -99.07

        # lngmin = 19.275
        # lngmax = 19.38

        # plt.contourf(lats, lngs, grid)
        tmp = tmp[latmin < tmp['lng']].dropna()
        tmp = tmp[tmp['lng'] < latmax].dropna()
        tmp = tmp[lngmin < tmp['lat']].dropna()
        tmp = tmp[tmp['lat'] < lngmax].dropna()

        #print(len(tmp))

        #print(tmp.to_numpy().shape)

        linear_regressor = LinearRegression()
        tmp['lng2'] = tmp['lng'] ** 2
        # tmp['latlng'] = tmp['lat'] * tmp['lng']
        tmp['lat2'] = tmp['lat'] ** 2
        tmp['one'] = 1
        # data = tmp[:, ['one', 'lat', 'lng', 'lng2', 'lat2', 'latlng']]
        data = tmp[['one', 'lat', 'lng', 'lng2', 'lat2']]
        # print(type(data))
        train = tmp[['metrostotales']]

        print(data.values.shape)
        print(train.values.shape)
        model = linear_regressor.fit(data.values, train.values)

        lats = np.linspace(latmin, latmax, 100)
        # lats = np.linspace(-60000.1422, -100000.2361, 100)
        lngs = np.linspace(lngmin, lngmax, 100)
        # lngs = np.linspace(-400000.8145, 100000.8181, 100)

        grid = np.meshgrid(lats, lngs)[0]
        # print(grid[0].shape)

        for x in range(len(lats)):
            for y in range(len(lngs)):
                lat = lats[x]
                lng = lngs[y]
                val = np.array([1, lat, lng, lng**2, lat**2])
                res = model.predict(val.reshape(1, -1))
                val = res[0, 0]
                grid[x][y] = val

        plt.figure()
        plt.contourf(lats, lngs, grid)
        lala = tmp[latmin < tmp['lng']].dropna()
        lala = lala[lala['lng'] < latmax].dropna()
        lala = lala[lngmin < lala['lat']].dropna()
        lala = lala[lala['lat'] < lngmax].dropna()
        print(len(lala))
        plt.scatter(lala['lng'], lala['lat'], c=lala['metrostotales'])
        plt.show()
