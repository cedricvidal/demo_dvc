import pandas
import pickle
from sklearn import linear_model
from sklearn import model_selection

X = pandas.read_csv("train/features.csv")
y = pandas.read_csv("train/target.csv")

model = linear_model.LinearRegression()
model.fit(X, y)

with open("model.pickle", "wb") as fd:
    pickle.dump(model, fd)
