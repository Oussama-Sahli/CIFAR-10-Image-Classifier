# -*- coding: utf-8 -*-


# src/train.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# 1️⃣ Charger le dataset CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2️⃣ Classes du dataset
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 3️⃣ Normaliser les images (valeurs entre 0 et 1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 4️⃣ Aperçu de quelques images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i])
    plt.title(classes[y_train[i][0]])
    plt.axis('off')
plt.show()

print(f"Dimensions X_train : {X_train.shape}")
print(f"Dimensions X_test : {X_test.shape}")
print(f"Dimensions y_train : {y_train.shape}")
print(f"Dimensions y_test : {y_test.shape}")
