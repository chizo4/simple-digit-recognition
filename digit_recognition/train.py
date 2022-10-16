'''
src/digit_recognition/train.py 

Train a simple Neural Network model based on MNIST datset.

Author: Filip J. Cierkosz (2022)
'''


import tensorflow as tf 


# Load MNIST datset.
mnist = tf.keras.datasets.mnist

# Preparing test and training data (x : image, y : class label (digit)).
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Specify model type, layers and connect neurons from each level.
model = tf.keras.models.Sequential() 
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Output layer - how likely is the image a particular digit (0-9)?
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile model with the optimizer and loss function. 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with and save...
model.fit(x_train, y_train, epochs=20)
model.save('digit.model')
