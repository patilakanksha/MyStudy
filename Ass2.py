# Assignment 2


#importing necessary libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#import dataset and split into train and test data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(len(x_train))
print(len(x_test))

print(x_train.shape)
print(x_test.shape)

plt.matshow(x_train[0])

#normalize the images by scaling pixel intensities to the range 0,1
x_train = x_train / 255
x_test = x_test / 255

#Define the network architecture using Keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # Flatten for compress the input
    keras.layers.Dense(128, activation='relu'), # Dence insures that each nuron of prev connected to next nuron
    keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

# compile the model
model.compile(optimizer='sgd',  #scochestic gradient decent
              loss='sparse_categorical_crossentropy', # it saves time
              metrics=['accuracy'])

# train the model
history=model.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss=%.3f" %test_loss)
print("Loss=", test_loss)
print("Accuracy=%.3f" %test_acc)

#Making prediction on new data
n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()

#we use predict() on new data
predicted_value=model.predict(x_test)
print("Handwritten number in the image is= %d" %np.argmax(predicted_value[n]))

# Plot graph for accuracy and loss
history.history
print(history.history.keys())

# model's Accuracy
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# model's loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss and accuracy')
plt.ylabel('accuracy/Loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy','loss','val_loss'])
plt.show()