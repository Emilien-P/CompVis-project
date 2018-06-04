import pandas as pd
import os
import random

labels = pd.read_csv("labels.csv")
# print(labels)

os.system("mkdir test_images train_images")
for i in range(0, 555):
	os.system("mkdir " + str(os.path.join("test_images", str(i) )))
	os.system("mkdir " + str(os.path.join("train_images", str(i) )))



for idx, row in labels.iterrows():
	if random.uniform(0, 1) < .1:
		# send to test
		imgpath = row.iloc[0]
		destpath = os.path.join("test_images", str(row.iloc[1]))
		os.system("mv " + imgpath + " " + destpath)
	else:
		imgpath = row.iloc[0]
		destpath = os.path.join("train_images", str(row.iloc[1]))
		os.system("mv " + imgpath + " " + destpath)

