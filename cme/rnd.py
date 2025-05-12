# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# Chargement des données depuis un fichier CSV
data = pd.read_csv('dossiers.csv')

# Encodage des colonnes catégoriques en valeurs numériques
encoders = {}

for col in ['Typologie du marché', 'Garantie', 'Situation fiscale', 'Fournisseur blacklisté', 'Visa (Oui/Non)']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Séparation des caractéristiques (features) et de la cible (target)
X = data.drop(columns=['Visa (Oui/Non)'])  # Toutes les colonnes sauf 'Visa (Oui/Non)'
y = data['Visa (Oui/Non)']  # Colonne cible

# Division des données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Vous pouvez ajuster n_estimators
rf.fit(X_train, y_train)

# Évaluation du modèle
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle Random Forest et des encodeurs dans un fichier
joblib.dump({'model': rf, 'encoders': encoders}, 'modele_rf_complet.pkl')
print("Modèle Random Forest et encodeurs sauvegardés sous le nom 'modele_rf_complet.pkl'.")

# Chargement du modèle sauvegardé pour utilisation future (par exemple, dans un autre script)
modele_charge = joblib.load('modele_rf_complet.pkl')
print("Modèle chargé avec succès.")
