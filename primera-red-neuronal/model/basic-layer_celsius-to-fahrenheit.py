import tensorflow as tf
# Essential for numerical arrays
import numpy as np
import os
# No CUDA GPU available, use CPU instead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Variables
celsius = np.array([-40, -10, 0, 8, 15 , 22 ,38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


# Keras is a high-level API that provides a clean and simple way to build and train neural networks.
# Dense layer is a fully connected layer.
layer = tf.keras.layers.Dense(units=1, input_shape=[1])
# Sequential model for this basic neural network.
model = tf.keras.Sequential([layer])

# Compile the model.
model.compile(
  # Adjust the learning rate for weights and biases to minimize the loss.
  optimizer = tf.keras.optimizers.Adam(0.1),
  # A smaller loss is better than a larger loss.
  loss='mean_squared_error' 
)


# Train the model.
print('Start model training...')
# Input, output, and how many times to repeat
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
# Finish training.
print('Model training finished!')


# Show the model's accuracy and loss over time in a graph.
import matplotlib.pyplot as plt
plt.xlabel('# Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()


# Predict the temperature for a new value.
print('Start model prediction test...')
result =  model.predict([[97.0]])
# Expected output [212.0] | Actual output [211.74287]
print('Result is ' + str(result) + ' Fahrenheit')
print('Model prediction test finished!')


# Show the model's internal weights and biases.
print('Model variables')
# Output -> Weight: [1.7982196] | Bias: [31.920912]
print(layer.get_weights())


# Result
# - - -
# Formula: F = C * 1.8 + 32
# Our neural network calculate a linear function:
# 100 celsius * 1.7982196 weight + 31.920912 bias = 211.74287 fahrenheit

# * Export the model
model.save('basic-layer_celsius-to-fahrenheit.h5')