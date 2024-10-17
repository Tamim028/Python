import pandas as pd
from matplotlib import pyplot as plt

#STEP 1: DATA READING AND UNDERSTANDING

df = pd.read_csv('data/images_analyzed_productivity.csv')
print(df.head())


#PLot productivity values to see the split between Good and Bad
#sizes = df['Productivity'].value_counts(sort = 1)
#plt.pie(sizes, shadow=True, autopct='%1.1f%%')



#STEP 2: DROP IRRELEVANT DATA
#In our example, Images_Analyzed reflects whether it is good analysis or bad
#so should not include it. ALso, User number is just a number and has no inflence
#on the productivity, so we can drop it.
#Productivity is created based on Images_Analyzed >= 17
df.drop(['Images_Analyzed'], axis=1, inplace=True)
df.drop(['User'], axis=1, inplace=True)


#STEP 3: Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value. 


#STEP 4: Convert non-numeric to numeric, if needed.
#Sometimes we may have non-numeric data, for example batch name, user name, city name, etc.
#e.g. if data is in the form of YES and NO then convert to 1 and 2

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 2
print(df.head())


#STEP 5: PREPARE THE DATA.

#Y is the data with dependent variable
Y = df["Productivity"].values
#Convert Y to int
Y=Y.astype('int')

#X is data with independent variables, everything except Productivity column
# Drop label column from X as you don't want that included as one of the features
X = df.drop(labels = ["Productivity"], axis=1)  
#print(X.head())

#STEP 6: SPLIT THE DATA into TRAIN AND TEST data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)


#STEP 7: Defining the model and training.
from sklearn.linear_model import LogisticRegression   #Import the relevant model
model = LogisticRegression()  #Create an instance of the model.

model.fit(X_train, y_train)  # Train the model using training data

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_result = model.predict(X_test)

from sklearn import metrics
#Print the prediction accuracy
print ("Event: Accuracy = ", metrics.accuracy_score(y_test, prediction_result))
#Test accuracy for various test sizes and see how it gets better with more training data


#UNDERSTAND WHICH VARIABLES HAVE MOST INFLUENCE ON THE OUTCOME
# To get the weights of all the variables

print(model.coef_) #Print the coefficients for each independent variable. 
#But it is not clear which one corresponds to what.
#SO let us print both column values and coefficients. 
#.Series is a 1-D labeled array capable of holding any data type. 
#Default index would be 0,1,2,3... but let us overwrite them with column names for X (independent variables)
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("Event:  Weights for each variables is a follows...")
print(weights)

#+VE VALUE INDICATES THAT THE VARIABLE HAS A POSITIVE IMPACT