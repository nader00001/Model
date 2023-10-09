import joblib

# Charger le modèle SVM
svm_model = joblib.load('model.pkl')

# Exemple d'un échantillon de données pour la prédiction (remplacez par vos propres données)
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Assurez-vous que les données correspondent aux caractéristiques que le modèle attend

# Effectuer une prédiction
predicted_class = svm_model.predict(sample_data)

# Afficher la classe prédite
print(f'Classe prédite : {predicted_class}')
