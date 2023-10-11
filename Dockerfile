# Utilisez une image Docker préexistante avec scikit-learn
FROM python:3.8-slim

# Définissez le répertoire de travail
WORKDIR /app

# Copiez les fichiers requis dans le conteneur
COPY requirements.txt ./
COPY model.py ./

# Installez les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port où votre application Flask s'exécutera
EXPOSE 5000

# Commande pour lancer votre application Flask
CMD ["python", "model.py"]
