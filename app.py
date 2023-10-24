import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Chargement du modèle VGG19 pré-entraîné
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Remplacez cette liste par vos genres musicaux
music_genres = ['Genre1', 'Genre2', 'Genre3', 'Genre4', 'Genre5']

def classify_music_genre(csv_file):
    try:
        # Chargez les données à partir du fichier CSV
        data = pd.read_csv(csv_file)

        # Créez une liste pour stocker les prédictions de genre musical
        predicted_genres = []

        for i in range(len(data)):
            # Chargez une image (ou des données) de votre fichier CSV
            # Assurez-vous d'adapter cette partie
            # Par exemple, vous pouvez charger une image à partir d'un chemin de fichier
            # image_path = data['chemin_image'][i]  # Assurez-vous d'adapter cette partie
            # img = image.load_img(image_path, target_size=(224, 224))
            # x = image.img_to_array(img)
            # x = preprocess_input(x)
            # x = np.expand_dims(x, axis=0)

            # Utilisation du modèle VGG19 pour prédire le genre musical
            # predictions = model.predict(x)  # Remplacez 'x' par vos données préparées
            # predicted_class = np.argmax(predictions)

            # Pour l'exemple, supposons que predicted_class est calculé
            predicted_class = i % len(music_genres)  # Exemple simple

            # Obtenez le genre musical prédit à partir de l'indice de classe
            predicted_genre = music_genres[predicted_class]
            predicted_genres.append(predicted_genre)

        return predicted_genres
    except Exception as e:
        return str(e)

# Fonction pour afficher un graphe à barres
def plot_genre_distribution(predicted_genres):
    genre_counts = dict()

    for genre in predicted_genres:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(genres, counts)
    plt.xlabel("Genres Musicaux")
    plt.ylabel("Nombre de Prédictions")
    plt.title("Distribution des Genres Musicaux Prédits")
    plt.xticks(rotation=45)
    plt.show()

if __name__ == '__main__':
    csv_file = "features_3_sec.csv"  # Remplacez par le chemin de votre fichier CSV
    genres_predits = classify_music_genre(csv_file)
    print(f"Genres prédits : {genres_predits}")

    # Afficher le graphe à barres
    plot_genre_distribution(genres_predits)
