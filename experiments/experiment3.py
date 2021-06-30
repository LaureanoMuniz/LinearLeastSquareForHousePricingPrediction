from dataclasses import dataclass
from experiments.nombres import LATITUD,LONGITUD,TIPO_DE_PROPIEDAD,ONE
import experiments.filter as filter
import experiments.geo as geo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import math

# import build.metnum as metnum
import sklearn.linear_model as metnum


def init_gaussianas(distx,disty):
	tamano_x_cuadrado = distx/cantidad_Gaussianas
	tamano_y_cuadrado = disty/cantidad_Gaussianas

	Gaussianas = []
	sigma_x = distx/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio))
	sigma_y = disty/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio))


	for i in range(cantidad_Gaussianas):
		for j in range(cantidad_Gaussianas):
			mu_x = tamano_x_cuadrado * (i + 1/2)
			mu_y = tamano_y_cuadrado * (j + 1/2)
			Gaussianas.append(Gaussiana(mu_x,mu_y,sigma_x,sigma_y))

	return Gaussianas

def turn_lat_long_into_XY(df,ciudad):
	def info(row):
		x = geo.distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_lo,row[LONGITUD])
		y = geo.distance(ciudad.lat_lo,ciudad.lng_lo,row[LATITUD],ciudad.lng_lo)
		row['X'] = x
		row['Y'] = y
		return row
	return df.apply(info, axis=1) 

def turn_propiedad_into_bin(df):
	def info(row):
		propiedadbin = row[TIPO_DE_PROPIEDAD] == "Casa"
		row['propiedadbin'] = int(propiedadbin)
		return row
	return df.apply(info, axis=1)

def matriz_de_gaussianas_aplicadas(df,Gaussianas):
	out = np.zeros((len(df),len(Gaussianas)))
	for i,funcion in enumerate(Gaussianas):
		out[:,i] = funcion.apply_vectorized(df['X'].tolist(),df['Y'].tolist())
	return out

def plot_mapita(predictor,Gaussianas,distx,disty,df):
		Xs = np.linspace(0,distx, 100)
		Ys = np.linspace(0,disty, 100)
		print(Xs)
		print(Ys)

		grid = np.meshgrid(Xs, Ys)[0]
        # print(grid[0].shape)

		for i,x in enumerate(Xs):
			for j,y in enumerate(Ys):
				val = np.zeros((1,len(Gaussianas) + 1))
				for k,funcion in enumerate(Gaussianas):
					val[0,k] = (funcion.apply(x,y))
				val[0,len(Gaussianas)] = 1
				res = predictor.predict(val)[0]
				if(res > 1):
					res =  1
				if(res < 0):
					res = 0
				grid[i][j] = res

		plt.figure()
		plt.contourf(Xs, Ys, grid, levels = np.linspace(0,1,1000))
		#plt.scatter(df['Y'], df['X'], c=df['propiedadbin'])
		plt.show()	

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
		exponent = ((x-self.Mu_x)/self.sigma_x)**2 + ((y-self.Mu_y)/self.sigma_y)**2
		try:
			return math.exp(-exponent)
		except:
			print("error, exponente es:")
			print(exponent)
			sys.exit()


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
	distx = geo.distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_hi,ciudad.lng_lo)
	disty = geo.distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_lo,ciudad.lng_hi)
	#nos movemos en el plano [0,distx]; [0,dist_y]. No estoy seguro si hacia falta
	
	Gaussianas = init_gaussianas(distx,disty)

	df[ONE] = 1
	df = filter.filter_city(df, ciudad)
	df = df[[LATITUD,LONGITUD,TIPO_DE_PROPIEDAD,ONE]]
	df = turn_lat_long_into_XY(df,ciudad)
	df = turn_propiedad_into_bin(df)

	n = len(df)
	train = df[:n*8//10]
	test = df[n*8//10:]

	one_reshaped_train = np.reshape(train[ONE].to_numpy(), newshape = (len(train),1))
	one_reshaped_test = np.reshape(test[ONE].to_numpy(), newshape = (len(test),1))	
	mean_squares_train = np.concatenate((matriz_de_gaussianas_aplicadas(train,Gaussianas),one_reshaped_train),axis=1)
	mean_squares_test = np.concatenate((matriz_de_gaussianas_aplicadas(test,Gaussianas),one_reshaped_test),axis=1)

	regressor = metnum.LinearRegression()
	regressor.fit(mean_squares_train,train['propiedadbin'])
	predicted = regressor.predict(mean_squares_test)
	print(predicted)
	plot_mapita(regressor,Gaussianas,distx,disty,df)

	##Todo esto deberian ser funciones en experiments.metricas?
	mean_predictor = np.full(len(test),np.mean(test['propiedadbin'].to_numpy()))
	error = predicted - test['propiedadbin'].to_numpy()
	error_de_la_media = mean_predictor - test['propiedadbin'].to_numpy()
	error = math.sqrt((error**2).sum() / len(test))
	error_de_la_media = math.sqrt((error_de_la_media**2).sum() / len(test)) 
	print(1 - (error / error_de_la_media))




def experimento3(df):
    for ciudad in ciudades:
        para_ciudad(ciudad, df)
