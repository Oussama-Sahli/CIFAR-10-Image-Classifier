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


#------------------------------------------------------------------------------
#Construire le CNN


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1️⃣ Encoder les labels en one-hot vectors
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2️⃣ Créer le modèle CNN
model = Sequential()

# Première couche convolution + pooling
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Deuxième couche convolution + pooling
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten pour connecter aux couches Dense
model.add(Flatten())

# Couche fully connected
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Couche de sortie
model.add(Dense(10, activation='softmax'))

# 3️⃣ Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4️⃣ Afficher le résumé du modèle
model.summary()



#------------------------------------------------------------------------------
#Entraîner le modèle et visualiser les performances



# 1️⃣ Définir les paramètres d'entraînement
epochs = 20
batch_size = 64

# 2️⃣ Entraîner le modèle
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# 3️⃣ Évaluer le modèle sur le test set
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nAccuracy sur le test : {test_acc:.4f}")
print(f"Loss sur le test : {test_loss:.4f}")

# 4️⃣ Visualiser l'accuracy et la loss
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()



# --------------------------------------------------------------------------
# Ajouter un drop out 
# Le Dropout aide à éviter l’overfitting → il “éteint” aléatoirement des neurones pendant l’entraînement.


from tensorflow.keras import layers, models

def create_model(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential()

    # Bloc 1
    model.add(layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.Conv2D(32, (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Bloc 2
    model.add(layers.Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Bloc 3
    model.add(layers.Conv2D(128, (3,3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

"""
On ajoute plus de convolutions (32 → 64 → 128 filtres).
Ça permet au modèle d’apprendre des features plus complexes (contours fins, textures, objets).
 Avec plus de couches + Dropout → on aura un modèle plus puissant mais mieux régularisé.

"""


from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisation (entre 0 et 1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# =====================
# 3. Data Augmentation
# =====================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)



# =====================
# 4. Entraînement
# =====================
batch_size = 64
epochs = 30

model = create_model()

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)
)


# =====================
# 5. Évaluation finale
# =====================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f" Test Accuracy: {test_acc:.4f}")
print(f" Test Loss: {test_loss:.4f}")



#-----------------------------------------------------------------------------
# Accuracy Loss train

import matplotlib.pyplot as plt

# Tracer Accuracy
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Tracer Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss pendant l\'entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()





#-----------------------------------------------------------------------------
# Tester le modèle sur des images individuelles 


import numpy as np
import matplotlib.pyplot as plt

# Classes CIFAR-10
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Choisir une image aléatoire dans le jeu de test
idx = np.random.randint(0, x_test.shape[0])
image = x_test[idx]
label = y_test[idx][0]

# Ajouter une dimension batch pour le modèle
img_input = np.expand_dims(image, axis=0)

# Faire la prédiction
pred = model.predict(img_input)
pred_class = np.argmax(pred, axis=1)[0]

# Afficher l'image avec la prédiction
plt.imshow(image)
plt.title(f"Vrai: {classes[label]} - Prédit: {classes[pred_class]}")
plt.axis('off')
plt.show()


#---------------------------------------------------------------------------
# Sauvegarder le modèle entier (architecture + poids + compilation)
import os

# Créer le dossier models s'il n'existe pas
if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/cifar10_model.h5")
print("Modèle entier sauvegardé dans models/cifar10_model.h5")


#-----------------------------------------------------------------------------
# performances

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Accuracy et Loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy pendant l’entraînement')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss pendant l’entraînement')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Matrice de confusion
y_pred = np.argmax(model.predict(x_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



#-----------------------------------------------------------------------------
# test sur images individuelles : 
    
import random
idx = random.randint(0, len(x_test)-1)
img = x_test[idx]
plt.imshow(img)
plt.show()

pred = np.argmax(model.predict(img[np.newaxis, ...]))
print(f"Classe prédite : {pred}, Classe réelle : {y_test[idx][0]}")





