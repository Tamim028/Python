#Definition: Logistic regression is a supervised machine learning algorithm used for classification tasks where 
#the goal is to predict the probability that an instance belongs to a given class or not (0 or 1, basically a binary classifier)

import pandas as pd
from matplotlib import pyplot as plt

#STEP 1: DATA READING AND UNDERSTANDING

df = pd.read_csv('data/images_analyzed_productivity.csv')
print(df.head())

df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


#STEP 3: Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value. 


#STEP 4: Convert non-numeric to numeric, if needed.

df.loc[df.Productivity == 'Bad', 'Productivity'] = 0
df.loc[df.Productivity == 'Good', 'Productivity'] = 1

print(df.head())


#STEP 5: PREPARE THE DATA.

#Y is the data with dependent variable
Y = df["Productivity"].values
#Convert Y to int
Y=Y.astype('int')


X = df.drop(labels = ["Productivity"], axis=1)  
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)


#STEP 7: Defining the model and training.
from sklearn.linear_model import LogisticRegression   #Import the relevant model
model = LogisticRegression()  #Create an instance of the model.
model.fit(X_train, y_train)  # Train the model using training data

prediction_result = model.predict(X_test)

#SAVING THE MODEL & LOADING the Model
import pickle
filename = "productivity_log_regression_model"
pickle.dump(model, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_test)








