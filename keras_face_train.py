__author__ = "Aaron Li"
# Adapted from the Keras Fashion Database Tutorial

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

folder = 'batch_1/'
train_images = np.load(folder+'train_batch_1.npy', allow_pickle=True)/255.0  # Scaling to values between 0.0 and 1.0
train_labels = np.load(folder+'train_labels_1.npy', allow_pickle=True)
test_images = np.load(folder+'test_batch_1.npy', allow_pickle=True)/255.0
test_labels = np.load(folder+'test_labels_1.npy', allow_pickle=True)

# Design the model
model = keras.Sequential([
    keras.layers.Conv2D(32, 5, input_shape=(32, 32, 3), padding='valid', activation=tf.nn.relu, name='conv1'),
    # 32 filters at kernel size 5
    keras.layers.Conv2D(64, 3, padding='valid', activation=tf.nn.relu, name='conv2'),
    # 64 filters at kernel size 3
    keras.layers.Conv2D(128, 3, padding='valid', activation=tf.nn.relu, name='conv3'),
    # 128 filters at kernel size 3
    keras.layers.Flatten(name='flatten1'),  # Flat layer
    keras.layers.Dense(1, activation=tf.nn.sigmoid, name='dense2')  # Sigmoid activation for output values from 0 to 1
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)

# Save the model
model.save('keras_face.h5')

# Generates label based on output value
def value2label(value):
    return 'Face' if value > 0.5 else 'Not Face'

# For plotting individual predictions
def plot_prediction(plot_index, predicted_value, true_value, img):
    predicted_label = value2label(predicted_value)
    true_label = value2label(true_value)

    if predicted_label != true_label:
        color = 'red'
    elif abs(predicted_value - true_value) > 0.05:
        color = 'blue'
    else:
        color = 'green'

    plt.subplot(num_rows, 2*num_cols, 2*plot_index+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.xlabel(predicted_label + " (" + true_label + ")",
               color=color)

    plt.subplot(num_rows, 2*num_cols, 2*plot_index+2)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.bar(0, predicted_value, color=color)
    plt.xlabel('{0:1.2f}'.format(predicted_value),
               color=color)
    plt.ylim([0, 1])

# Plot of 25 predictions
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    prediction, tru, im = predictions[i][0], test_labels[i], test_images[i]
    plot_prediction(i, prediction, tru, im)
plt.show()
