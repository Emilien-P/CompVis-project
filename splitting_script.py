import pandas as pd
import os

train_labels = pd.read_csv("meta/train.txt")
os.system("mkdir test_images train_images")
for idx, row in train_labels.iterrows():
	train_item = row.iloc[0]
	path = os.path.join("images", train_item)
	path = path + ".jpg"
	os.system("cp --parents " + path + " train_images")

test_labels = pd.read_csv("meta/test.txt")
for idx, row in test_labels.iterrows():
	test_item = row.iloc[0]
	path = os.path.join("images", test_item)
	path = path + ".jpg"
	os.system("cp --parents " + path + " test_images")
