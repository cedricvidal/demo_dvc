import pandas

df = pandas.read_csv("./dataset.csv")

features  = df["x"]
target = df["y"]

features.to_frame().to_csv("./features.csv", index=False)
target.to_frame().to_csv("./target.csv", index=False)

