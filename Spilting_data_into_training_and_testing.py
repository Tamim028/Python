import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model  

df = pd.read_csv('data/BCG_BD_Coverage.csv')

year_arr = df['Year'].values
vaccine_arr = df['Vaccine_Coverage'].values

x = year_arr.reshape(-1,1)
y = vaccine_arr.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.4, random_state=1)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train) #

predicted_values = model.predict(X_test)

print("Test size: ", len(Y_test))
print("Predicted size: ", len(predicted_values))

print("Error between Y_test and Predicted = ", np.mean(predicted_values - Y_test)**2)
plt.scatter(predicted_values, predicted_values-Y_test)
plt.hlines(y=0, xmin=0, xmax=200)
