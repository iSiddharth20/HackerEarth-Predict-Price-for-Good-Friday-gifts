#!/usr/bin/python3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib

'''
Volumes Prediction
'''
original = pd.read_csv('dataset.csv')
df_train = original.copy()
df_train['lsg_4'] = (df_train['lsg_4'] - df_train['lsg_4'].mean()) / df_train['lsg_4'].std()
df_train['lsg_4'] = np.c_[np.ones(df_train['lsg_4'].shape[0]), df_train['lsg_4']] 
features = ['price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']
target = 'volumes'
m = df_train[['volumes', 'price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']]
d = m.dropna()
X = d.drop(columns = [target], axis = 1)
Y = d[target]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 0)
final_conf = 0
final_est = 0
final_rs = 0
final_lr = 0
final_depth = 0
for est in range(120,151):
	for rs in range(1,6):
		for depth in range(3,7):
			lr = 0.2
			while(lr <= 0.6):
				model = GradientBoostingRegressor(n_estimators = est , random_state = rs , learning_rate = lr , max_depth = depth)
				model.fit(X_train,Y_train)
				Y_pred = model.predict(X_test)
				conf = round((r2_score(Y_test,Y_pred))*100,3)
				if (conf > 90 and conf > final_conf):
					final_conf = conf
					final_est = est
					final_rs = rs
					final_lr = lr
					final_depth = depth
					volume_model = model
				lr += 0.001
joblib.dump(volume_model,'Volumes.pkl')
df_train = original.copy()
test  = pd.DataFrame(df_train[df_train[target].isnull()])
df_train.dropna(inplace = True)
predictions = finalmodel.predict(test[features])
test[target] = predictions
frame = [df_train,test]
final = pd.concat(frame)
final.to_csv('newtrain.csv', index = False)

'''
Price Prediction
'''
df_train = final.copy()
features = ['volumes', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']
target = 'price'
m = df_train[['volumes', 'price', 'gift_type', 'gift_category', 'gift_cluster', 'lsg_1', 'lsg_2', 'lsg_3','lsg_4', 'lsg_5', 'lsg_6', 'is_discounted']]
d = m.dropna()
X = d.drop(columns = [target], axis = 1)
Y = d[target]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)
final_conf = 0
final_est = 0
final_rs = 0
final_lr = 0
final_depth = 0
for est in range(120,151):
	for rs in range(1,6):
		for depth in range(3,7):
			lr = 0.2
			while(lr <= 0.6):
				model = GradientBoostingRegressor(n_estimators = est , random_state = rs , learning_rate = lr , max_depth = depth)
				model.fit(X_train,Y_train)
				Y_pred = model.predict(X_test)
				conf = round((r2_score(Y_test,Y_pred))*100,3)
				if (conf > 90 and conf > final_conf):
					final_conf = conf
					final_est = est
					final_rs = rs
					final_lr = lr
					final_depth = depth
					volume_model = model
				lr += 0.001
joblib.dump(price_model,'BestModel.pkl')
