#!/usr/bin/python3

# Getting all the Required Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Getting the Dataset into the Program
original = pd.read_csv('dataset.csv')

'''
Since the Dataset has many Null Values in the 'Volumes' Column.
Null Values will be predicted taking Target Variable as a Feature.
Volumes Prediction (Column with Null Values)
'''

# Dataset Used is the Raw Dataset
df_train = original.copy()

# Normalizing Outliser using Z-Score
df_train['lsg_4'] = (df_train['lsg_4'] - df_train['lsg_4'].mean()) / df_train['lsg_4'].std()
df_train['lsg_4'] = np.c_[np.ones(df_train['lsg_4'].shape[0]), df_train['lsg_4']] 

# Identifing Features and Target Variable
features = ['price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']
target = 'volumes'

# Slicing the Data Frame to create Model
m = df_train[['volumes', 'price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']]
d = m.dropna()

# Data to Train the Model
X = d.drop(columns = [target], axis = 1)
Y = d[target]

# Splitting the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

# Creating the Model (Optimised)
model = GradientBoostingRegressor(n_estimators = 120 , random_state = 2 , learning_rate = 0.378 , max_depth = 5)

# Dividing the Dataframe to Impute predicted Null Values
df_train = original.copy()
test  = pd.DataFrame(df_train[df_train[target].isnull()])
df_train.dropna(inplace = True)

# Predicting Null Values in the 'Volumes' Column
predictions = model.predict(test[features])

# Imputing Predicted Null Values in the Dataset
test[target] = predictions
frame = [df_train, test]

# New Dataset with Predicted Null Values is Ready
final = pd.concat(frame)

'''
Since Null Valies have been predicted with High Accuracy,
We now have more data to train our Model.
Price Prediction (Original Target Variable)
'''

# Dataset Used is the one with Predicted Null Values
df_train = final.copy()

# Identifing Features and Target Variable
features = ['volumes', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']
target = 'price'

# Slicing the Data Frame to create Model
m = df_train[['volumes', 'price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']]
d = m.dropna()

# Data to Train the Model
X = d.drop(columns = [target], axis = 1)
Y = d[target]

# Splitting the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

# Creating the Model (Optimised)
model = GradientBoostingRegressor(n_estimators=135 , random_state=18 , learning_rate=0.359 , max_depth = 3)

# Model exported ad a Pickle File 
joblib.dump(model, 'BestPrice.pkl')
