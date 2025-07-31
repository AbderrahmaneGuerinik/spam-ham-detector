# Étape 1 : Utiliser une image officielle Python 3.10
FROM python:3.10

# Étape 2 : Créer un dossier de travail
WORKDIR /app

# Étape 3 : Copier tous les fichiers dans l'image
COPY . .

# Étape 4 : Installer les dépendances
RUN pip install --upgrade pip
RUN pip install --prefer-binary -r requirements.txt
RUN python -m spacy download en_core_web_sm


# Étape 5 : Exposer le port de Streamlit
EXPOSE 8501

# Étape 6 : Commande pour lancer Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
