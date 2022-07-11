import tensorflow as tf
import tensorflow_datasets as tfds
import os
# No CUDA GPU available, use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Use Zalando's Fashion-MNIST dataset
data, metadata = tfds.load('fashion_mnist', as_supervised = True, with_info = True)
print(metadata)

# Split the dataset into train and test
train_data, test_data = data['train'], data['test']

# Categories names
categories = metadata.features['label'].names
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(categories)


# * Normalize data from 0-255 to 0-1
def normalizer(images, labels):
  images = tf.cast(images, tf.float32)
  # Divide by 255 to normalize
  images /= 255
  return images, labels


# Normalize dataset
train_data = train_data.map(normalizer)
test_data = test_data.map(normalizer)

# Add data to cache memory to speed up training process instead of reading from disk
train_data = train_data.cache()
test_data = test_data.cache()

# Show the first image from the train dataset
for image, label in test_data.take(1):
  break
# Reshape the image to a 28x28 image
image = image.numpy().reshape((28, 28))

import matplotlib.pyplot as plt

# Show the first image
plt.figure()
plt.imshow(image, cmap = plt.cm.binary)
plt.colorbar()
plt.grid(False)
# plt.show()

# Show 25 images with labels
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_data.take(25)):
  image = image.numpy().reshape((28, 28))
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap = plt.cm.binary)
  plt.xlabel(categories[label])
# plt.show()

# * Build the model
model = tf.keras.Sequential([
  # Flatten the image to a 1-dimensional vector
  # 28 x 28 with 1 channel for grayscale
  tf.keras.layers.Flatten(input_shape = (28, 28, 1)),

  # Add 2 hidden layers with 50 neurons each and a ReLU activation function
  # ReLu function is the most common activation function and is used to avoid negative values
  tf.keras.layers.Dense(50, activation = tf.nn.relu),
  tf.keras.layers.Dense(50, activation = tf.nn.relu),

  # Add an output layer with 10 neurons and a softmax activation function
  # Softmax function will give us a probability distribution for each category
  tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

# * Compile the model
model.compile(
  optimizer = 'adam',
  # Loss function is the mean of the cross-entropy between the predicted and the actual labels
  loss = tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics = ['accuracy']
)

TRAIN_EXAMPLES_SIZE = metadata.splits['train'].num_examples
TEST_EXAMPLES_SIZE = metadata.splits['test'].num_examples
print(TRAIN_EXAMPLES_SIZE, TEST_EXAMPLES_SIZE)

# Batch size is the number of images we want to train on at once
BATCH_SIZE = 32

# Random images with batch size
train_data = train_data.repeat().shuffle(TRAIN_EXAMPLES_SIZE).batch(BATCH_SIZE)
test_data = test_data.batch(BATCH_SIZE)

# * Train the model

import math

history = model.fit(train_data, epochs = 5, steps_per_epoch = math.ceil(TRAIN_EXAMPLES_SIZE / BATCH_SIZE))

# Show loss throughout training
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
# plt.show()

# * Evaluate the model
# Test images and show the predicted labels and the real labels using blue for correct and red for incorrect

import numpy as np

for test_images, test_labels in test_data.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)

def draw_image(i, predictions_arr, real_labels, images):
  predictions_arr, real_label, image = predictions_arr[i], real_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(image[..., 0], cmap = plt.cm.binary)

  predicted_label = np.argmax(predictions_arr)

  if predicted_label == real_label:
    color = 'blue' # Correct prediction
  else:
    color = 'red' # Incorrect prediction
  
  plt.xlabel(
    '{} {:2.0f}% ({})'.format(
      categories[predicted_label],
      100 * np.max(predictions_arr),
      categories[real_label]
    ), color = color
  )

def draw_arr_value(i, predictions_arr, real_label):
  predictions_arr, real_label = predictions_arr[i], real_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  graph = plt.bar(range(10), predictions_arr, color='#777777')
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_arr)

  graph[predicted_label].set_color('red')
  graph[real_label].set_color('blue')

rows = 5
columns = 5
images_size = rows * columns
plt.figure(figsize = (2 * 2 * columns, 2 * rows))

# Show 25 images with labels and predictions
for i in range (images_size):
  plt.subplot(rows, 2 * columns, 2 * i + 1)
  draw_image(i, predictions, test_labels, test_images)
  plt.subplot(rows, 2 * columns, 2 * i + 2)
  draw_arr_value(i, predictions, test_labels)

plt.show()

# Take a random image from the test dataset and show the predicted label
random = np.random.choice(len(test_images))
image = test_images[random]
image = np.array([image])
real_label = test_labels[random]
prediction = model.predict(image)

print('Random Prediction')
print('Real label:', categories[real_label])
print('Predicted label:', categories[np.argmax(prediction)])

# * Export the model
model.save('basic_image_classifier.h5')