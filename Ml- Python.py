import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

# If using Google Colab

#from google.colab import drive
#drive.mount('/content/drive')
#df = pd.read_csv('/content/drive/MyDrive/car data.csv')
#df.head()

df = pd.read_csv('car data.csv')
df.head()
## lets check the shape
df.shape

## lets check the basic information of the dataset
df.info()

len(df[df.duplicated()])
## dropping duplicates
df.drop_duplicates(inplace=True)
## recheck
len(df[df.duplicated()])

## dropping the column name
df.drop('Car_Name',axis=1,inplace=True)

## substracting the year of purchase with the current year and extracting the age of the car
df['age_of_the_car'] = 2022 - df['Year']

## dropping the column year
df.drop('Year',axis=1,inplace=True)

df.head(2)
df['Fuel_Type'].unique()
array(['Petrol', 'Diesel', 'CNG'], dtype=object)

df['Seller_Type'].unique()
array(['Dealer', 'Individual'], dtype=object)

df['Transmission'].unique()
array(['Manual', 'Automatic'], dtype=object)

## Manual Encoding:
df['Fuel_Type'] = df['Fuel_Type'].replace({'Petrol':0, 'Diesel':1, 'CNG':2})

df['Seller_Type'] = df['Seller_Type'].replace({'Dealer':0, 'Individual':1})

df['Transmission'] = df['Transmission'].replace({'Manual':0, 'Automatic':1})

df['Fuel_Type'].unique()
array([0, 1, 2], dtype=int64)

df['Seller_Type'].unique()
array([0, 1], dtype=int64)

df['Transmission'].unique()
array([0, 1], dtype=int64)

## Final dataframe
df.head(2)

X = df.drop('Selling_Price',axis=1)
y = df['Selling_Price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
(209, 7) (90, 7)
(209,) (90,)

## Let us build simple random forest regressor model
rf = RandomForestRegressor()
rf.fit(X_train,y_train)

## Let us check the r2-score to see hows our model is performing

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

r2_train = r2_score(y_train,y_train_pred)
r2_test = r2_score(y_test,y_test_pred)

print('r2-score train:',r2_train)
print('r2-score test',r2_test)