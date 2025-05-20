import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib



# Chargement du modèle
objet = joblib.load('modele_svm_complet.pkl')
modele_charge = objet['model']
encoders = objet['encoders']
scaler = objet['scaler']
colonnes_numeriques = objet['colonnes_numeriques']

# Nouvelle donnée brute
nouvelle_donnee = pd.DataFrame([{
    'Typologie du marché': 'Travaux',
    'Montant du contrat': 5000000,
    'Garantie': 'Aucune',
    'Délai de réalisation': 5,
    'Expérience fournisseur': 7,
    'Nombre de projets similaires': 2,
    'Notation interne': 14,
    'Chiffre d\'affaire': 30000000,
    'Situation fiscale': 'Conforme',
    'Fournisseur blacklisté': 'Non'
}])

# Encodage des colonnes catégorielles
for col in ['Typologie du marché', 'Garantie', 'Situation fiscale', 'Fournisseur blacklisté']:
    nouvelle_donnee[col] = encoders[col].transform(nouvelle_donnee[col])

# Standardisation des colonnes numériques
nouvelle_donnee[colonnes_numeriques] = scaler.transform(nouvelle_donnee[colonnes_numeriques])

# Prédiction
prediction = modele_charge.predict(nouvelle_donnee)
print("Prédiction :", "Visa Accordé" if prediction[0] == 1 else "Visa Refusé")
