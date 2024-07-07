import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from tensorflow.math import confusion_matrix



(X_train, Y_train), (X_test, Y_test) =  mnist.load_data()

type(X_train)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


print(X_train[10])

print(X_train[10].shape)

plt.imshow(X_train[25])
plt.show()

print(Y_train[25])

print(Y_train.shape, Y_test.shape)

print(np.unique(Y_train))


print(np.unique(Y_test))


X_train = X_train/255
X_test = X_test/255


print(X_train[10])

model = keras.Sequential([
                          keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(50, activation='relu'),
                          keras.layers.Dense(10, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=10)

loss, accuracy = model.evaluate(X_test, Y_test)
print(accuracy)

print(X_test.shape)

plt.imshow(X_test[0])
plt.show()

print(Y_test[0])

Y_pred = model.predict(X_test)

print(Y_pred.shape)

print(Y_pred[0])

label_for_first_test_image = np.argmax(Y_pred[0])
print(label_for_first_test_image)


Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)


conf_mat = confusion_matrix(Y_test, Y_pred_labels)

print(conf_mat)

plt.figure(figsize=(15,7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')

input_image_path = '/content/MNIST_digit.png'

input_image = cv2.imread(input_image_path)

type(input_image)

print(input_image)

cv2_imshow(input_image)

input_image.shape

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

grayscale.shape

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize.shape

cv2_imshow(input_image_resize)

input_image_resize = input_image_resize/255

type(input_image_resize)

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

input_image_resize = cv2.resize(grayscale, (28, 28))

input_image_resize = input_image_resize/255

image_reshaped = np.reshape(input_image_resize, [1,28,28])

input_prediction = model.predict(image_reshaped)

input_pred_label = np.argmax(input_prediction)

print('The Handwritten Digit is recognised as ', input_pred_label)

