import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Chargement des données
data = pd.read_csv('dossiers.csv')

# Encodage des colonnes catégoriques
encoder = LabelEncoder()
data['Typologie du marché'] = encoder.fit_transform(data['Typologie du marché'])
data['Garantie'] = encoder.fit_transform(data['Garantie'])
data['Situation fiscale'] = encoder.fit_transform(data['Situation fiscale'])
data['Fournisseur blacklisté'] = encoder.fit_transform(data['Fournisseur blacklisté'])
data['Visa (Oui/Non)'] = encoder.fit_transform(data['Visa (Oui/Non)'])

# Séparation des caractéristiques (features) et de la cible (target)
X = data.drop(columns=['Visa (Oui/Non)']).values  # Convertir en tableau numpy
y = data['Visa (Oui/Non)'].values

# Normalisation des caractéristiques
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape pour le CNN (ajout d'une dimension pour le canal)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle CNN
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Activation sigmoïde pour une classification binaire
])

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision sur l'ensemble de test : {accuracy * 100:.2f}%")


# Prédictions sur l'ensemble de test
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()  # Convertir les probabilités en classes binaires

# Génération et affichage du rapport de classification
print(classification_report(y_test, y_pred))

# Prédictions sur de nouvelles données
nouvelle_donnee = np.array([[1, 5000000, 1, 6, 10, 2, 13, 45000000, 1, 0]])  # Exemple
nouvelle_donnee = scaler.transform(nouvelle_donnee)  # Normalisation
nouvelle_donnee = nouvelle_donnee.reshape(1, nouvelle_donnee.shape[1], 1)  # Reshape
prediction = model.predict(nouvelle_donnee)
print("Prédiction pour le dossier :", "Visa Accordé" if prediction[0] > 0.5 else "Visa Refusé")
