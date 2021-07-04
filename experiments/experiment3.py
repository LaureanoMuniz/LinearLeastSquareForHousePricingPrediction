from dataclasses import dataclass
from experiments.nombres import LATITUD,LONGITUD,TIPO_DE_PROPIEDAD,ONE
import experiments.filter as filter
import experiments.geo as geo
from experiments.kfold import kfold
import experiments.métricas as metricas
from experiments.cities import ciudades
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

import build.metnum as metnum


def init_gaussianas(distx,disty,times = 1):
	tamano_x_cuadrado = distx/cantidad_Gaussianas
	tamano_y_cuadrado = disty/cantidad_Gaussianas

	Gaussianas = []
	sigma_x = distx/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio)) * times
	sigma_y = disty/(2*cantidad_Gaussianas * math.sqrt(-ln_un_medio)) * times


	for i in range(cantidad_Gaussianas):
		for j in range(cantidad_Gaussianas):
			mu_x = tamano_x_cuadrado * (i + 1/2)
			mu_y = tamano_y_cuadrado * (j + 1/2)
			Gaussianas.append(Gaussiana(mu_x,mu_y,sigma_x,sigma_y))

	return Gaussianas

def turn_lat_long_into_XY(df,ciudad, distx, disty):
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

def plot_mapita(predictor,Gaussianas,distx,disty,df,nombreCiudad):
		Xs = np.linspace(0,distx, 100)
		Ys = np.linspace(0,disty, 100)
		
		grid = np.meshgrid(Xs, Ys)[0]
        
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

		mapaCiudad = plt.imread(f'data/{nombreCiudad}.png')
		BBox = (np.min(Xs), np.max(Xs), np.min(Ys), np.max(Ys))
		
		plt.figure(nombreCiudad)
		plt.imshow(mapaCiudad, extent=BBox, aspect='equal', alpha=1)
		plt.contourf(Xs, Ys, grid, levels =[0,0.49999,0.500001,1], alpha = 0.35)
		plt.xticks([])
		plt.yticks([])
		plt.show()

		plt.figure(nombreCiudad)
		plt.imshow(mapaCiudad, extent=BBox, aspect='equal', alpha=1)
		plt.contourf(Xs, Ys, grid, levels =[0,0.49999,0.500001,1], alpha = 0.35)
		plt.scatter(df['X'], df['Y'], c=df['propiedadbin'], s = 3)
		plt.xticks([])
		plt.yticks([])
		plt.show()

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

ln_un_medio = -0.69314718 

cantidad_Gaussianas = 25
	

def para_ciudad(ciudad, df):
	disty = geo.distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_hi,ciudad.lng_lo)
	distx = geo.distance(ciudad.lat_lo,ciudad.lng_lo,ciudad.lat_lo,ciudad.lng_hi)
	#nos movemos en el plano [0,distx]; [0,dist_y]. No estoy seguro si hacia falta
	
	Gaussianas = init_gaussianas(distx,disty,3)

	df = filter.filter_city(df, ciudad)
	df = df[[LATITUD,LONGITUD,TIPO_DE_PROPIEDAD,ONE]]
	df = turn_lat_long_into_XY(df,ciudad,distx, disty)
	df = turn_propiedad_into_bin(df)

	precisiones = []
	mapita = True
	proporcionescasas = []
	print(len(df))
	
	for train, test in kfold(df):
		print("new test")
		one_reshaped_train = np.reshape(train[ONE].to_numpy(), newshape = (len(train),1))
		one_reshaped_test = np.reshape(test[ONE].to_numpy(), newshape = (len(test),1))	
		mean_squares_train = np.concatenate((matriz_de_gaussianas_aplicadas(train,Gaussianas),one_reshaped_train),axis=1)
		mean_squares_test = np.concatenate((matriz_de_gaussianas_aplicadas(test,Gaussianas),one_reshaped_test),axis=1)

		regressor = metnum.LinearRegression()
		regressor.fit(mean_squares_train,train['propiedadbin'])
		proporcionescasas.append(np.mean(train['propiedadbin'].to_numpy()))
		predicted = regressor.predict(mean_squares_test)
		for i in range(np.shape(predicted)[0]):
			predicted[i] = max(predicted[i],0)
			predicted[i] = min(predicted[i],1)
		if(mapita):
			plot_mapita(regressor,Gaussianas,distx,disty,df,ciudad.nombre)
			mapita = False
		precisiones.append(metricas.precision_casas(test['propiedadbin'].to_numpy(),predicted))
		print("anduvo precision")
	print("precision")
	print(np.mean(precisiones))
	print("proporciones casas")
	print(np.mean(proporcionescasas))


def experimento3(df):
    #for ciudad in ciudades:
        #para_ciudad(ciudad, df)
	para_ciudad(ciudades[1],df)