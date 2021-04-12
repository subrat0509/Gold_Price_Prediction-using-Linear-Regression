#Importing necessary packages & libraries
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import scipy
import matplotlib
import matplotlib.pyplot as plt


#Dataset quick info and making sure there are no null values
df = pd.read_csv('gld_price_data.csv')
df.head()
df.info()



#Correlations matrix and heatmap
corr = df.corr()
plt.figure(figsize = (6,5))
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values, annot = True, fmt = '.2f', linewidths = 0.30)
plt.title('df correlation heatmap', y = 1.12, size = 13, loc = "center")
plt.show()
#Print the correlation score
print(corr['GLD'].sort_values(ascending = False), '\n')




#Predict the GLD variable value based on other variables
df.hist(bins = 50, figsize = (15,10))
plt.show()

sns.pairplot(df.loc[:,df.dtypes == 'float64'])
plt.show()


#Preparing a copy to work with
df["new"]=df["SLV"]*5
df.head()
df1 = df.copy()
temp = df1[['SPX','USO','SLV','EUR/USD','new']]
x = temp.iloc[:, :].values
y = df1.iloc[:, 2].values

#Splitting the train and test dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Training the model with the linear regression function
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)



#Checking the training and test set accuracy
y_pred = regressor.predict(x_test)
accuracy_train = regressor.score(x_train, y_train)
accuracy_test = regressor.score(x_test, y_test)
print("Training Accuracy: ", accuracy_train)
print("Testing Accuracy: ", accuracy_test)

#Error checking for regression
from sklearn import metrics
print('MAE :'," ", metrics.mean_absolute_error(y_test,y_pred))
print('MSE :'," ", metrics.mean_squared_error(y_test,y_pred))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#Visualising the result
plt.plot(y_test, color = 'blue', label = 'Real value')
plt.plot(y_pred, color = 'deeppink', label = 'Predicted value')
plt.grid(True)
plt.title('Final result')
plt.xlabel('Oberservations')
plt.ylabel('GLD')
plt.legend()
plt.show()
