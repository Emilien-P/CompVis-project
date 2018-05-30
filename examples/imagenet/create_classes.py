import pandas as pd

df = pd.read_csv("food-101/meta/labels.txt")
text = "("

for idx, elem in df.iterrows():
	label = elem.iloc[0]
	text = text + "\"" + label + "\","

with open("labels_tuple.txt", "w") as file:
	file.write(text)
