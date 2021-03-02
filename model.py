import json 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import joblib

# Loading the dataset 
df = pd.read_csv("/home/ghost/Desktop/Processor/Processor/server/Dataset/data.csv", skipinitialspace=True)
x_cols = [c for c in df.columns if c != 'income']
#set input matrix and target column
X = df[x_cols]
y = df['income']
# show first rows of data
# print(df.head())

#checking for statistical info
#print(df.describe())

# Checking for general info 
# print(df.info())
# print(df.head())
# data split train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1234)
# fill missing values
train_mode = dict(X_train.mode().iloc[0])
X_train = X_train.fillna(train_mode)
print(train_mode)
# convert categoricals
encoders = {}
for column in ['workclass', 'education', 'marital-status',
                'occupation', 'relationship', 'race',
                'sex','native-country']:
    categorical_convert = LabelEncoder()
    X_train[column] = categorical_convert.fit_transform(X_train[column])
    encoders[column] = categorical_convert

# train the Random Forest algorithm
rf = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rf = rf.fit(X_train, y_train)
# train the Extra Trees algorithm
et = ExtraTreesClassifier(n_estimators = 100)
et = et.fit(X_train, y_train)

print(rf)
# save preprocessing objects and RF algorithm
# joblib.dump(train_mode, "./train_mode.joblib", compress=True)
# joblib.dump(encoders, "./encoders.joblib", compress=True)
# joblib.dump(rf, "./random_forest.joblib", compress=True)
# joblib.dump(et, "./extra_trees.joblib", compress=True)