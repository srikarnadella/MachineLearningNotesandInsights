import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

path = r"H:\CodingProjects\MLCourse\Product\cars.xls"
df = pd.read_excel(path)

#Extracts data that we want to use (relation between mileage and price for cost)
df1=df[['Mileage','Price']]

#breaks up data into chunks into 10,000 mile chunks between 50,000 
bins =  np.arange(0,50000,10000)

#Groups then matches the mileage to the bins
groups = df1.groupby(pd.cut(df1['Mileage'],bins)).mean()

#Graph here models milage vs price
#groups['Price'].plot.line()
#plt.show()

#Using this to bring the data together by normalizing it, typically improves performance
scale = StandardScaler()


#X Extracts the 3 features, predicts price using these 3 features (not including make and model since you can't mix ordinal and numerical in this model)
X = df[['Mileage', 'Cylinder', 'Doors']]

#What we are trying to predict
y = df['Price']

#Pass features through the scaler to normalize it
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values)

#Constant column for the b (y-intercept) values 
#Adds column of 1's at the end
X = sm.add_constant(X)

#Creates and trains the model
est = sm.OLS(y, X).fit()


#Provides details on the training
#https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
print(est.summary())




#How to use the model to make a prediction

#Example car to predict value of
scaled = scale.transform([[45000, 8, 4]])

#Need to add that constant column in again.
scaled = np.insert(scaled[0], 0, 1) 

#Using the model we made to predict it using the scaled values
predicted = est.predict(scaled)
print(predicted)