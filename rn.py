import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Charger les données
df = pd.read_csv('dossiers_fictifs.csv')

# 2. Préparer les données

# Encoder les colonnes catégorielles
colonnes_a_encoder = ['Typologie du marché', 'Garantie', 'Situation fiscale', 'Fournisseur blacklisté']
encoders = {}  # Dictionnaire pour stocker les encodeurs

for col in colonnes_a_encoder:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Sauvegarder l'encodeur

# Encoder la cible (Visa Oui/Non)
df['Visa (Oui/Non)'] = df['Visa (Oui/Non)'].map({'Non': 0, 'Oui': 1})

# Séparer les variables explicatives (X) et la cible (y)
X = df.drop(['Dossier ID', 'Visa (Oui/Non)'], axis=1)
y = df['Visa (Oui/Non)']

# Standardiser les données (très important pour le réseau de neurones)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Construire le modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 1 seule sortie entre 0 et 1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entraîner le modèle
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# 5. Évaluer sur le test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Précision sur l'ensemble de test : {test_acc:.2f}")

# 6. Fonction pour prédire un nouveau dossier
def predire_visa(nouveau_dossier):
    # nouveau_dossier = dictionnaire
    df_nouveau = pd.DataFrame([nouveau_dossier])
    
    # Encoder les colonnes avec les encodeurs sauvegardés
    for col in colonnes_a_encoder:
        if col in df_nouveau.columns:
            df_nouveau[col] = encoders[col].transform(df_nouveau[col])
        else:
            raise ValueError(f"La colonne '{col}' est manquante dans le dossier.")

    # Standardiser les données
    df_nouveau_scaled = scaler.transform(df_nouveau)

    # Prédiction du modèle
    prob = model.predict(df_nouveau_scaled)[0][0]
    pourcentage = prob * 100
    decision = 'Oui' if prob >= 0.5 else 'Non'
    return pourcentage, decision

# Exemple d'utilisation
nouveau = {

    'Typologie du marché': 'Fournitures',
    'Montant du contrat': 5000000,
    'Garantie': 'Aucune',
    'Délai de réalisation': 4,
    'Expérience fournisseur': 5,
    'Nombre de projets similaires': 2,
    'Notation interne': 15,
    'Chiffre d\'affaire': 30000000,
    'Situation fiscale': 'Conforme',
    'Fournisseur blacklisté': 'Oui'
}

pourcentage, decision = predire_visa(nouveau)
print(f"Visa : {decision} ({pourcentage:.2f}%)")
