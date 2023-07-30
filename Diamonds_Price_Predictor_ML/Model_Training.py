import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score as evs

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

# Dataset Import
df = pd.read_csv('Diamonds Prices2022.csv')


# EDA & Preprocessing
print(df.info())
print(df.dtypes)

for c in df.columns:
    if df[c].dtype == 'object':
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

print(df)
print(df.dtypes)


# Train Test Split
df = df.drop('Unnamed: 0', axis= 1)

features = df.drop('price', axis= 1)
target = df['price']

print(features, target)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.25, shuffle= True, random_state= 42)


# Model Training
models = [DecisionTreeRegressor(), Ridge(), LinearRegression(), RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy : \n{evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy : \n{evs(Y_test, pred_test)}\n')

    