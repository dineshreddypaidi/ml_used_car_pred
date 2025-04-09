import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("../data/processed/data.csv")

X = data.drop(columns=['selling_price']).values

y = data['selling_price'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.05)

model = RandomForestRegressor()

model.fit(X_train,y_train)

joblib.dump(model, f"../models/model.pkl")