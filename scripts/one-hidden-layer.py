from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import csv

# === CONSTANTS ===
data_path = "../data/"
results_path = "../results/"
train_path = data_path + "training.csv"
test_path = data_path + "test.csv"
IdLookupTable_path = results_path + "IdLookupTable.csv"
output_file_path = results_path + "submission.csv"

validation_proportion = 0.1

keypoint_names = ["left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y","left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x","left_eye_outer_corner_y","right_eye_inner_corner_x","right_eye_inner_corner_y","right_eye_outer_corner_x","right_eye_outer_corner_y","left_eyebrow_inner_end_x","left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y","right_eyebrow_inner_end_x","right_eyebrow_inner_end_y","right_eyebrow_outer_end_x","right_eyebrow_outer_end_y","nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y","mouth_right_corner_x","mouth_right_corner_y","mouth_center_top_lip_x","mouth_center_top_lip_y","mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"]
keypoint_indices = dict((keypoint_name, index) for index, keypoint_name in enumerate(keypoint_names))

image_size = 96
num_keypoints = 15
max_pixel_value = 255

input_size = image_size*image_size
num_hidden = 100
output_size = 2*num_keypoints

learning_rate = 1e-3
dropout_keep_prob = 0.5

num_epochs = 200
batch_size = 100
display_step = 10

plot = True
generate_results = True

flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19), (22, 24), (23, 25)]

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
	X = X / (float(max_pixel_value)/2)
	
	if train:
		# Get Y (keypoints coordinates)
		Y = df[df.columns[:-1]].values
		
		# Normalize Y
		Y = (Y - image_size/2) / (float(image_size)/2)
		
		# Shuffle X and Y
		permutation = np.random.permutation(len(X))
		X = X[permutation]
		Y = Y[permutation]
	else:
		Y = None
	
	return X, Y	

print("Constructing training set...")
train_X, train_Y = get_data_set(train_path, train=True)

valid_index = int(len(train_X) * validation_proportion)

valid_X = train_X[:valid_index]
train_X = train_X[valid_index:]
valid_Y = train_Y[:valid_index]
train_Y = train_Y[valid_index:]

print("Constructing test set...")
test_X, _ = get_data_set(test_path, train=False)

print("\nTrain X")
print(train_X.shape)
print("train_X[0]:",train_X[0])

print("\nTrain Y")
print(train_Y.shape)
print("train_Y[0]:",train_Y[0])

print("\nValid X")
print(valid_X.shape)
print("valid_X[0]:",valid_X[0])

print("\nValid Y")
print(valid_Y.shape)
print("valid_Y[0]:",valid_Y[0])

print("\nTest X")
print(test_X.shape)
print("test_X[0]:",test_X[0])

# === MODEL ===
def new_weights(shape, xavier=True):
	dev = 1.0
	if xavier and len(shape) == 2:
		dev = 1/shape[0]
	initial = tf.random_normal(shape, stddev=dev)
	return tf.Variable(initial)
	
def new_biases(shape, xavier=True):
	dev = 1.0
	if xavier and len(shape) == 2:
		dev = 1/shape[0]
	initial = tf.random_normal(shape, stddev=dev)
	return tf.Variable(initial)
	
def simple_linear_layer(input,shape):
	assert (len(shape) == 2),"Shape : [input,output]"
	weights = new_weights(shape)
	biases = new_biases([shape[-1]])
	logits = tf.matmul(input, weights) + biases
	return logits
	
def simple_relu_layer(input,shape,dropout_keep_prob=None):
	logits = simple_linear_layer(input,shape)
	logits = tf.nn.relu(logits)
	if not dropout_keep_prob is None:
		logits = tf.nn.dropout(logits, dropout_keep_prob)
	return logits
	
def one_hidden_layer_model(input,dropout_keep_prob=None):
	with tf.name_scope('hidden_layer'):
		hidden_logits = simple_relu_layer(input,[input_size,num_hidden],dropout_keep_prob=keep_prob)
	
	with tf.name_scope('output_layer'):
		output_logits = simple_linear_layer(hidden_logits,[num_hidden,output_size])
		
	return output_logits

print ("\nConstructing model...")

with tf.name_scope('placeholders'):
	pixels = tf.placeholder(tf.float32, shape=[None, input_size])
	keypoints = tf.placeholder(tf.float32, shape=[None, output_size])
	keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('model'):
	predicted_keypoints = one_hidden_layer_model(pixels,dropout_keep_prob=keep_prob)
	
with tf.name_scope('loss'):
	loss = tf.sqrt(tf.reduce_mean(tf.square(predicted_keypoints - keypoints)))
	
with tf.name_scope('Train_step'):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
init = tf.initialize_all_variables()
	
# === TRAINING MODEL ===
def seconds2minutes(time):
	minutes = int(time) // 60
	seconds = int(time) % 60
	return minutes, seconds
	
with tf.Session() as session:
	session.run(init)
	
	num_examples = len(train_X)
	num_steps_per_epoch = num_examples//batch_size
	
	absolute_step = 0
	
	steps = []
	train_losses = []
	valid_losses = []
	
	print("\nSTART TRAINING (",num_epochs,"epochs,",num_steps_per_epoch,"steps per epoch )")
	begin_time = time_0 = time.time()
	
	for epoch in range(num_epochs):
		print("*** EPOCH",epoch,"***")
		for step in range(num_steps_per_epoch):
			batch_X = train_X[step * batch_size:(step + 1) * batch_size]
			batch_Y = train_Y[step * batch_size:(step + 1) * batch_size]
			_ = session.run(train_step, feed_dict={pixels: batch_X, keypoints: batch_Y, keep_prob: dropout_keep_prob})
			
			absolute_step += 1
			
			if step % display_step == 0:
				train_loss = session.run(loss, feed_dict={pixels: batch_X, keypoints: batch_Y, keep_prob: 1.0})
				valid_loss = session.run(loss, feed_dict={pixels: valid_X, keypoints: valid_Y, keep_prob: 1.0})
			
				print("Batch Loss =",train_loss,"at step",absolute_step)
				print("Validation Loss =",valid_loss,"at step",absolute_step)
				print("Estimated Score =",valid_loss*image_size/2,"at step",absolute_step)
				
				steps.append(absolute_step)
				train_losses.append(train_loss)
				valid_losses.append(valid_loss)
				
				# Time spent is measured
				if absolute_step > 0:
					t = time.time()
					d = t - time_0
					time_0 = t
					
					print("Time:",d,"s to compute",display_step,"steps")
					
		last_batch_X = train_X[num_examples * batch_size:]
		last_batch_Y = train_Y[num_examples * batch_size:]
		_ = session.run(train_step, feed_dict={pixels: last_batch_X, keypoints: last_batch_Y, keep_prob: dropout_keep_prob})
		absolute_step += 1
	
	total_time = time.time() - begin_time
	total_time_minutes, total_time_seconds = seconds2minutes(total_time)
	print("*** Total time to compute",num_epochs,"epochs:",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")
	
	# === TEST ===
	test_predictions = []
	num_test_steps = len(test_X)//batch_size
	print('\n*** Start testing (',num_test_steps,'steps ) ***')
	for step in range(num_test_steps):
		batch_X = test_X[step * batch_size:(step + 1) * batch_size]
		pred = session.run(predicted_keypoints, feed_dict={pixels : batch_X, keep_prob: 1.0})
		test_predictions.extend(pred)
		
	last_batch_X = test_X[num_test_steps * batch_size:]
	pred = session.run(predicted_keypoints, feed_dict={pixels : last_batch_X, keep_prob: 1.0})
	test_predictions.extend(pred)
		
	test_predictions = np.array(test_predictions)
	test_predictions = test_predictions * image_size/2 + image_size/2
	print('Test prediction',test_predictions.shape)

# === PLOTTING ===	
if plot:
	print("Plotting...")
	import matplotlib.pyplot as plt
	plt.plot(steps, train_losses, 'ro', steps, valid_losses, 'bs')
	x1,x2,y1,y2 = plt.axis()
	# plt.axis((x1,x2,0,50))
	plt.show()
	
# === GENERATE SUBMISSION FILE ===
def get_predictions_indices(IdLookupTable,keypoint_indices,imageId):
	feature_names = IdLookupTable[IdLookupTable["ImageId"] == imageId]["FeatureName"].tolist()
	return [keypoint_indices[feature_name] for feature_name in feature_names]

if generate_results:
	print("Reading IdLookupTable...")
	IdLookupTable = pd.read_csv(IdLookupTable_path)

	print('Generating submission file...')
	with open(output_file_path, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		writer.writerow(['RowId','Location'])
		rowid = 1
		for i in range(len(test_predictions)):
			imageId = i + 1
			predictions = test_predictions[i]
			for j in get_predictions_indices(IdLookupTable,keypoint_indices,imageId):
				prediction = predictions[j]
				if prediction < 0:
					prediction = 0
				elif prediction > image_size:
					prediction = image_size
				writer.writerow([rowid,prediction])
				rowid += 1
		print('Results saved to',output_file_path)