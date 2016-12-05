from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf

# === CONSTANTS ===
data_path = "../data/"
train_path = data_path + "training.csv"
test_path = data_path + "test.csv"

# === GET DATA ===
def get_data_set(path, train=True):
	# Read data from csv file
	df = pd.read_csv(path)
	
	# Drop incomplete rows
	df = df.dropna()
	
	# Get X (image pixels)
	# Image: Convert string values to numpy arrays
	df['Image'] = df['Image'].apply(lambda image: np.fromstring(image, sep=' '))
	
	# Stack all arrays into a numpy array and normalize
	X = np.vstack(df['Image'].values)
	
	# Normalize X
	X = (X  - 128) / float(128)
	
	if train:
		# Get Y (keypoints coordinates)
		Y = df[df.columns[:-1]].values
		
		# Normalize Y
		Y = (Y - 48) / 48
		
		# Shuffle X and Y
		permutation = np.random.permutation(len(X))
		X = X[permutation]
		Y = Y[permutation]
	else:
		Y = None
	
	return X, Y

print("Constructing training set...")
train_X, train_Y = get_data_set(train_path, train=True)

print("Constructing test set...")
test_X, _ = get_data_set(test_path, train=False)

print("\nTrain X")
print(train_X.shape)

print("\nTrain Y")
print(train_Y.shape)

print("\nTest X")
print(test_X.shape)