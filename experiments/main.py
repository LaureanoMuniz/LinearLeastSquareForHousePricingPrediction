from experiments.nombres import PRECIO
# from experiments.parser import embellish
# import build.metnum as metnum
import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# import experiments.filter as filter
# import experiments.geo as geo

import experiments.experimento1
import experiments.experiment5
import experiments.experiment4
from experiments.precio_m2 import precio_m2

import experiments.nombres as nombres

# from sklearn.linear_model import LinearRegression

# df = pd.read_csv('data/train.csv')
# df = filter.filter(df)
# df = embellish(df)
# del df[nombres.TÍTULO]
# del df[nombres.DESCRIPCIÓN]
df = pd.read_csv('data/parsed.csv')
df = df.sample(frac=1)
df[nombres.ONE] = 1
print(df[PRECIO].max())
df = precio_m2(df)
experiments.experiment5.experiment5(df)
# experiments.experimento1.experimento1(df, True)
# experiments.experiment4.experiment4(df)

# ciudades = df['ciudad'].value_counts(sort=True)[:10].index.tolist()
# print(ciudades)
