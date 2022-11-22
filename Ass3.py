import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test,y_test) = datasets.cifar10.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

print(y_train[:5])

y_train = y_train.reshape(-1,)   #2D array can be reshaped into 1D array using reshape(-1)
print(y_train[:5])

y_test = y_test.reshape(-1,)
print(y_test[:5])

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(x, y, index):
    plt.figure(figsize = (20,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])

plot_sample(x_train, y_train, 0)
plot_sample(x_train, y_train, 1)
plot_sample(x_train, y_train, 4)

#Normalizing data
x_train = x_train / 255.0
x_test = x_test / 255.0

#Define network architecture of your model
annmodel = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(1000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

#compile model using optimizer
annmodel.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

annmodel.fit(x_train, y_train, epochs=5)

print(annmodel.evaluate(x_test,y_test))


# # define cnn network architecture
# cnnmodel = models.Sequential([
#     layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
#
# cnnmodel.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# history=cnnmodel.fit(x_train, y_train,validation_data=(x_test,y_test),epochs=10)
#
# print(cnnmodel.evaluate(x_test,y_test))
#
# plt.plot(history.history['accuracy'],label='acc', color='red')
# plt.plot(history.history['val_accuracy'],label='val_acc', color='green')
# plt.legend()
#
# y_pred = cnnmodel.predict(x_test)
# y_pred[:5]