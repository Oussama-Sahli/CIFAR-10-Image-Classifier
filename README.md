# CIFAR-10 Image Classifier

##  Description
Ce projet consiste à construire un modèle de **classification d'images** sur le dataset **CIFAR-10**.  
Le modèle est basé sur un **Convolutional Neural Network (CNN)** avec plusieurs blocs de convolution, max pooling, dropout et data augmentation pour améliorer la performance et réduire l’overfitting.

---

##  Contenu du projet
- `src/train.py` : Script principal pour charger les données, créer le modèle, entraîner et évaluer.
- `models/` : Contient le modèle sauvegardé (`cifar10_weights.h5`).
- `results/` : Contient les visualisations et la matrice de confusion.
- `README.md` : Ce fichier.

---

##  Technologies et bibliothèques utilisées
- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib

---

##  Étapes réalisées

1. **Chargement des données**  
   - Dataset CIFAR-10 : 50 000 images d'entraînement, 10 000 images de test.
   - Normalisation des images entre 0 et 1.

2. **Création du modèle CNN**
   - Bloc 1 : 2 convolutions 32 filtres + MaxPooling + Dropout
   - Bloc 2 : 2 convolutions 64 filtres + MaxPooling + Dropout
   - Bloc 3 : 2 convolutions 128 filtres + MaxPooling + Dropout
   - Dense : 512 neurones + Dropout + couche finale softmax 10 classes

3. **Data Augmentation**
   - Rotation aléatoire, translation horizontale/verticale, flip horizontal, zoom.

4. **Entraînement**
   - 30 epochs, batch_size = 64

5. **Évaluation**
   - Accuracy finale : ~0.75
   - Loss finale : ~0.75
   - Matrice de confusion générée pour visualiser les prédictions

6. **Sauvegarde du modèle**
   - Modèle sauvegardé dans `models/cifar10_weights.h5` pour utilisation future.

---

##  Résultats
- Modèle capable de reconnaître 10 classes différentes : avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion.
- Accuracy test : ~0.75
- Loss test : ~0.75

---

##  Instructions pour exécuter le projet

1. Cloner le repo :
```bash
git clone https://github.com/Oussama-Sahli/CIFAR-10-Image-Classifier.git
