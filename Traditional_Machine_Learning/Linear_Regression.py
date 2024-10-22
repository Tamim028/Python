import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import linear_model  

df = pd.read_csv('data/BCG_BD_Coverage.csv')

year_arr = df['Year'].values
vaccine_arr = df['Vaccine_Coverage'].values

X_train = year_arr.reshape(-1,1)
Y_train = vaccine_arr.reshape(-1,1)

model = linear_model.LinearRegression().fit(X_train, Y_train)

p = model.predict([[2024]])

print("Linear Regression Prediction ", p)
  