from dataclasses import dataclass
#from . import nombres
#from experiments import filter
#from experiments import geo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import math

# import build.metnum as metnum
import sklearn.linear_model as metnum

###import a mano

def distance(lat1, lon1, lat2, lon2):
    p = math.pi/180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p) *\
        math.cos(lat2*p) * (1-math.cos((lon2-lon1)*p))/2

    if a < 0:
        a = 0
    if math.sqrt(a) > 1 or math.sqrt(a) < 0:
        print(a)

    a = min(1, math.sqrt(a))
    a = max(0, a)
    return 12742 * math.asin(a)

LATITUD = "lat"
LONGITUD = "lng"
METROS_TOT = "metrostotales"
METROS_CUB = "metroscubiertos"
PRECIO = "precio"
HABITACIONES = "habitaciones"
BAÑOS = "banos"
GARAJES = "garages"
TÍTULO = "titulo"
DESCRIPCIÓN = "descripcion"
GIMNASIO = "gimnasio"
USOS_MULT = "usosmultiples"
PILETA = "piscina"
ESCUELAS = "escuelascercanas"
COMERCIOS = "centroscomercialescercanos"
SEGURIDAD = "seguridad"
SALAS = "salas"
NATURALEZA = "naturaleza"
CHETOS = "chetos"
PRECIO_M2 = "precio_por_m2"
PRECIO_M2_NEAREST = "precio_por_m2_nearest"
TIPO_DE_PROPIEDAD = "tipodepropiedad"
ONE = "one"
eps = 1e-6

def filter_city(df, ciudad):
    in_city = (df[LATITUD] > ciudad.lat_lo)\
        & (df[LATITUD] < ciudad.lat_hi)\
        & (df[LONGITUD] > ciudad.lng_lo)\
        & (df[LONGITUD] < ciudad.lng_hi)

    return df[in_city]

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

df = pd.read_csv('data/train.csv')
df = df.sample(frac=1)
df = filter(df)

###fin del import a mano


def turn_lat_long_into_XY(df,ciudad):
	def info(row):
		x = distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_lo,row[LONGITUD])
		y = distance(ciudad.lat_lo,ciudad.lng_lo,row[LATITUD],ciudad.lng_lo)
		row['X'] = x
		row['Y'] = y
		return row
	return df.apply(info, axis=1) 



@dataclass
class Ciudad:
    nombre: str
    lat_lo: np.float64
    lat_hi: np.float64
    lng_lo: np.float64
    lng_hi: np.float64


@dataclass
class Gaussiana:
	Mu_x: np.float64
	Mu_y: np.float64
	sigma_x: np.float64
	sigma_y: np.float64 		##estamos asumiendo matrices de covarianza con x,y independientes

	def apply(self,x,y):
		#try:
			exponent = ((x-self.Mu_x)/self.sigma_x)**2 + ((y-self.Mu_y)/self.sigma_y)**2
			try:
				return math.exp(-exponent)
			except:
				print("error, exponente es:")
				print(exponent)
				sys.exit()
			'''except:
			print("error, x and y are:")
			print(x)
			print(y)
			print("and Mu_x and Mu_y are:")
			print(self.Mu_x)
			print(self.Mu_y)
			print("and sigma_x and sigma_y are:")
			print(self.sigma_x)
			print(self.sigma_y)
			sys.exit()'''

	def apply_vectorized(self,x,y):
		result = np.zeros(len(x))
		for i in range(len(x)):
			result[i] = self.apply(x[i],y[i])
		return result

ciudades = [
    Ciudad(
        nombre='Guadalajara',
        lat_lo=20.573875,
        lat_hi=20.767892,
        lng_lo=-103.496963,
        lng_hi=-103.199645,
    )
]


ln_un_medio = -0.69314718 

cantidad_Gaussianas = 25



def para_ciudad(ciudad, df):
	distx = distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_hi,ciudad.lng_lo)
	disty = distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_lo,ciudad.lng_hi)
	#nos movemos en el plano [0,distx]; [0,dist_y]. No estoy seguro si hacia falta
	tamano_x_cuadrado = distx/cantidad_Gaussianas
	tamano_y_cuadrado = disty/cantidad_Gaussianas


	Gaussianas = []
	sigma_x = distx/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio))
	sigma_y = disty/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio))


	for i in range(cantidad_Gaussianas):
		mu_x = tamano_x_cuadrado * (i + 1/2)
		mu_y = tamano_y_cuadrado * (i + 1/2)
		Gaussianas.append(Gaussiana(mu_x,mu_y,sigma_x,sigma_y))
	
	df = filter_city(df, ciudad)
	df = df[[LATITUD,LONGITUD,TIPO_DE_PROPIEDAD]]
	df = turn_lat_long_into_XY(df,ciudad)

	n = len(df)
	train = df[:n*8//10]
	test = df[n*8//10:]

	mean_squares_train = np.zeros((len(train),cantidad_Gaussianas))
	for i,funcion in enumerate(Gaussianas):
		#try:
			mean_squares_train[:,i] = funcion.apply_vectorized(train['X'].tolist(),train['Y'].tolist())

		#except:
			#print("error, iteration is:")
			#print(i)

para_ciudad(ciudades[0],df)