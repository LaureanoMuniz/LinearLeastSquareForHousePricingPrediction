from pandas.io.parsers import read_csv
from experiments.nombres import PRECIO
from experiments.parser import embellish
import build.metnum as metnum
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import experiments.filter as filter
import experiments.geo as geo

import experiments.experimento1
import experiments.experiment5
import experiments.experiment4
from experiments.precio_m2 import precio_m2

import experiments.nombres as nombres

# from sklearn.linear_model import LinearRegression
import sys


try:
    df= pd.read_csv('data/parsed.csv')

except FileNotFoundError: 
    df = pd.read_csv('data/train.csv')
    df = filter.filter(df)
    df = embellish(df)
    del df[nombres.TÍTULO]
    del df[nombres.DESCRIPCIÓN]
    df.to_csv('data/parsed.csv')


df = df.sample(frac=1) #Mezclar el dataset
df[nombres.ONE] = 1 # Cambio a one por un 1.


class Experimentos:
    def case_1():
        return experiments.experimento1.experimento1(df, False)

    def case_2():
        experiments.experimento1.experimento1(df, True)

    def case_3():
        return 

    def case_4():
        experiments.experiment4.experiment4(df)

    def case_5():
        experiments.experiment5.experiment5(df)



try:
    func = (getattr(Experimentos,'case_' + str(sys.argv[1])))
    func()
except:
    print('MODO DE USO: main.py NUMERO_DE_EXPERIMENTO')



# experiments.experimento1.experimento1(df, True)
# experiments.experiment4.experiment4(df)
# ciudades = df['ciudad'].value_counts(sort=True)[:10].index.tolist()
# print(ciudades)
