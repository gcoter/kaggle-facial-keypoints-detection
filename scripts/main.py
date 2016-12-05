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
output_file_path = results_path + "submission.csv"

validation_proportion = 0.1

image_size = 96
num_keypoints = 15
max_pixel_value = 255

input_size = image_size*image_size
num_hidden = 100
output_size = 2*num_keypoints

learning_rate = 1e-3

num_epochs = 20
batch_size = 100
display_step = 10

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
	X = (X  - max_pixel_value/2) / (float(max_pixel_value)/2)
	
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
def new_weights(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)
	
def new_biases(shape):
	initial = tf.random_normal(shape)
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
	
def one_hidden_layer_model(input):
	with tf.name_scope('hidden_layer'):
		hidden_logits = simple_relu_layer(input,[input_size,num_hidden])
	
	with tf.name_scope('output_layer'):
		output_logits = simple_linear_layer(hidden_logits,[num_hidden,output_size])
		
	return output_logits

print ("\nConstructing model...")

with tf.name_scope('placeholders'):
	pixels = tf.placeholder(tf.float32, shape=[None, input_size])
	keypoints = tf.placeholder(tf.float32, shape=[None, output_size])

with tf.name_scope('model'):
	predicted_keypoints = one_hidden_layer_model(pixels)
	
with tf.name_scope('loss'):
	loss = tf.sqrt(tf.reduce_mean(tf.square(predicted_keypoints - keypoints)))
	
with tf.name_scope('Train_step'):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	
init = tf.initialize_all_variables()
	
# === TRAINING MODEL ===
def seconds2minutes(time):
	minutes = int(time) / 60
	seconds = int(time) % 60
	return minutes, seconds
	
with tf.Session() as session:
	session.run(init)
	
	num_examples = len(train_X)
	num_steps_per_epoch = num_examples/batch_size
	
	print("\nSTART TRAINING (",num_epochs,"epochs,",num_steps_per_epoch,"steps per epoch )")
	begin_time = time_0 = time.time()
	
	for epoch in range(num_epochs):
		print("*** EPOCH",epoch,"***")
		for step in range(num_steps_per_epoch):
			batch_X = train_X[step * batch_size:(step + 1) * batch_size]
			batch_Y = train_Y[step * batch_size:(step + 1) * batch_size]
			_, loss_value = session.run([train_step,loss], feed_dict={pixels: batch_X, keypoints: batch_Y})
			absolute_step = epoch * num_steps_per_epoch + step
			
			if step % display_step == 0:
				valid_loss = session.run(loss, feed_dict={pixels: valid_X, keypoints: valid_Y})
			
				print("Batch Loss =",loss_value,"at step",absolute_step)
				print("Validation Loss =",valid_loss,"at step",absolute_step)
				
				# Time spent is measured
				if absolute_step > 0:
					t = time.time()
					d = t - time_0
					time_0 = t
					
					print("Time:",d,"s to compute",display_step,"steps")
					
		last_batch_X = train_X[num_examples * batch_size:]
		last_batch_Y = train_Y[num_examples * batch_size:]
		_, loss_value = session.run([train_step,loss], feed_dict={pixels: last_batch_X, keypoints: last_batch_Y})
	
	total_time = time.time() - begin_time
	total_time_minutes, total_time_seconds = seconds2minutes(total_time)
	print("*** Total time to compute",num_epochs,"epochs:",total_time_minutes,"minutes and",total_time_seconds,"seconds (",total_time,"s)***")
	
	# === TEST ===
	test_predictions = []
	num_test_steps = len(test_X)/batch_size
	print('\n*** Start testing (',num_test_steps,'steps ) ***')
	for step in range(num_test_steps):
		batch_X = test_X[step * batch_size:(step + 1) * batch_size]
		pred = session.run(predicted_keypoints, feed_dict={pixels : batch_X})
		test_predictions.extend(pred)
		
	last_batch_X = test_X[num_test_steps * batch_size:]
	pred = session.run(predicted_keypoints, feed_dict={pixels : last_batch_X})
	test_predictions.extend(pred)
		
	test_predictions = np.array(test_predictions)
	test_predictions = test_predictions * image_size/2 + image_size/2
	print('Test prediction',test_predictions.shape)
	
# === GENERATE SUBMISSION FILE ===
print('Generating submission file...')
with open(output_file_path, 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['RowId','Location'])
	rowid = 1
	for i in range(len(test_predictions)):
		predictions = test_predictions[i]
		for j in range(len(predictions)):
			writer.writerow([rowid,predictions[j]])
			rowid += 1
	print('Results saved to',output_file_path)