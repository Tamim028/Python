import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("data/images_analyzed_productivity.csv")
print(df.head())


#STEP 2: DROP IRRELEVANT DATA
df.drop(['Images_Analyzed', 'User'], axis=1, inplace=True)


#STEP 3: Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value. 


#STEP 4: Convert non-numeric to numeric, if needed.
df.loc[df.Productivity == 'Bad', 'Productivity'] = 0
df.loc[df.Productivity == 'Good', 'Productivity'] = 1

print(df.head())


#STEP 5: PREPARE THE DATA.

Y = df["Productivity"].values 
Y=Y.astype('int')

X = df.drop(labels = ["Productivity"], axis=1)  


#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


#STEP 7: Defining the model and training.

from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_estimators = 10, random_state = 30) #Instantiate model with 10 decision trees
model.fit(X_train, y_train)


#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)

from sklearn import metrics

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data // test_size=0.2 or 0.4

#feature importance..
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


