import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


df=pd.read_csv("dataset_34_with_header.csv")

#removes rows that have 31 nan => 305 - 31= 274 no nan
df5 = df.dropna(thresh=274)

#removes columns that have over 50% of values missing
df6 = df5.dropna(thresh=0.5*len(df), axis=1)

#reset the index without adding it as a column
df7 = df6.reset_index(drop=True)

#selecting columns that have only 0 and 1 values
df8 = df7[df7.columns[df7.isin([0,1]).all()]]

#check for duplicates
df9 = len(df7) - df7.nunique()

#replacing nan with the median
df10 = df7.fillna(df.median())

#checking that all nan have been replaced and that there are no nan
df11 = df10.isna().mean().round(4) * 100

#check for negative values
df12 = np.sum((df11 < 0).values.ravel())

#compute logarithm
df13 = np.log(df10)

#remove -inf from computed logarithm transforming it to nan
df14 = df13.replace(-np.inf, np.nan)

#passing the nan back to the median
df15 = df14.fillna(df.median())

#checking that there are no nan
df16 = df15.isna().mean().round(4) * 100

#computing the z-score
df17 = (df15 - df15.mean()) / df15.std()

#passing the nan back to the median
df18 = df17.fillna(df.median())

#checking that there are no nan
df19 = df18.isna().mean().round(4) * 100

#df18 is the final dataset

#we first separate the DV from the IV

# separating y to another DataFrame
y = df17.iloc[:, -1]

#these are the predictor variables
x = df17.drop(['y'], axis=1)
x = x.fillna(df.median())

#histogram of y
plt.title("Histogram of Y",fontsize=15)
plt.hist(df17['y'],bins=20)
plt.show()

# linear regression
reg=linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
reg.fit(x_train, y_train)

a= reg.predict(x_test)

b= np.mean((a-y_test)**2)


#we use polynomial regression
x = np.array(x)
y = np.array(y)

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)
