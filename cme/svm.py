import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib

# Chargement des données
data = pd.read_csv('dossiers.csv')

# Colonnes catégorielles à encoder
colonnes_cat = [
    'Typologie du marché',
    'Garantie',
    'Situation fiscale',
    'Fournisseur blacklisté',
    'Visa (Oui/Non)'  # Cible
]

# Encodage des colonnes catégorielles
encoders = {}
for col in colonnes_cat:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Séparation X / y
X = data.drop(columns=['Visa (Oui/Non)'])
y = data['Visa (Oui/Non)']

# Colonnes numériques à standardiser
colonnes_numeriques = [
    'Montant du contrat', 'Délai de réalisation', 'Expérience fournisseur',
    'Nombre de projets similaires', 'Notation interne', 'Chiffre d\'affaire'
]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardisation
scaler = StandardScaler()
X_train[colonnes_numeriques] = scaler.fit_transform(X_train[colonnes_numeriques])
X_test[colonnes_numeriques] = scaler.transform(X_test[colonnes_numeriques])

# Entraînement du modèle SVM
svm = SVC(kernel='rbf', C=2, gamma='scale', class_weight='balanced', probability=True)

svm.fit(X_train, y_train)

# Évaluation
y_pred = svm.predict(X_test)
print("=== Rapport de classification ===")
print(classification_report(y_test, y_pred))
# Sauvegarde du modèle uniquement
joblib.dump(svm, 'modele_svm.pkl')
print("Modèle SVM sauvegardé.")
# Sauvegarde des encodeurs et du scaler dans un autre fichier
joblib.dump({
    'encoders': encoders,
    'scaler': scaler
}, 'encodeurs_scaler.pkl')
print("Encodeurs et scaler sauvegardés.")

# Sauvegarde du modèle, des encodeurs et du scaler
#joblib.dump({
#    'model': svm,
 #   'encoders': encoders,
  #  'scaler': scaler,
   # 'colonnes_numeriques': colonnes_numeriques
#}, 'modele_svm_complet.pkl')
#print("Modèle, encodeurs et scaler sauvegardés.")
