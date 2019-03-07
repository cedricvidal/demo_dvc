import pandas
import os
from sklearn import model_selection

features = pandas.read_csv("./features.csv")
target = pandas.read_csv("./target.csv")

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, train_size=0.8)

for d in ("train", "test"):
    if not os.path.exists(d):
        os.mkdir(d)

X_train.to_csv("train/features.csv", index=False)
y_train.to_csv("train/target.csv", index=False)
X_test.to_csv("test/features.csv", index=False)
y_test.to_csv("test/target.csv", index=False)
