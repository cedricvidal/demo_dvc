import pandas
import pickle

X = pandas.read_csv("test/features.csv")
y = pandas.read_csv("test/target.csv")

with open("model.pickle", "rb") as fd:
    model = pickle.load(fd)

r2 = model.score(X, y)

with open("metric.txt", "w") as fd:
    fd.write("r2: {:.2f}\n".format(r2))
