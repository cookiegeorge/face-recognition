__author__ = "Aaron Li"
# Adapted from the Keras Fashion Database Tutorial

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# Test image batch
# Redirect as desired to a nx32x32x3 and nx1 npy object respectively
folder = 'batch_1/'
test_images = np.load(folder+'test_batch_1.npy', allow_pickle=True)/255.0
test_labels = np.load(folder+'test_labels_1.npy', allow_pickle=True)

# Recall saved model
model = keras.models.load_model('keras_face.h5')

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Run model on test batch
predictions = model.predict(test_images)

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
