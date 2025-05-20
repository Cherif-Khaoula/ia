import matplotlib.pyplot as plt
import seaborn as sns
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

# Extraire les importances des caractéristiques
importances = rf.feature_importances_
features = X.columns  # Assurez-vous que X est le DataFrame des données d'entraînement

# Créer un DataFrame pour les afficher facilement
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Visualiser l'importance des caractéristiques
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importance des caractéristiques dans le modèle Random Forest')
plt.show()
from sklearn.metrics import roc_curve, auc

# Calculer les probabilités de prédiction
y_prob = rf.predict_proba(X_test)[:, 1]

# Calculer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Afficher la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc='lower right')
plt.show()

from sklearn.model_selection import learning_curve

# Obtenir les courbes d'apprentissage
train_sizes, train_scores, test_scores = learning_curve(rf, X, y, cv=5, n_jobs=-1)

# Calculer la moyenne et l'écart type des scores d'entraînement et de test
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

# Visualiser la courbe d'apprentissage
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Score d\'entrainement', color='blue')
plt.plot(train_sizes, test_mean, label='Score de test', color='red')
plt.title('Courbe d\'apprentissage')
plt.xlabel('Taille de l\'ensemble d\'entraînement')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Liste des différents nombres d'arbres à tester
n_estimators_list = [10, 50, 100, 200, 300, 500]

# Initialisation des résultats
scores = []

# Boucle pour entraîner et tester le modèle avec différents nombres d'arbres
for n_estimators in n_estimators_list:
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)  # Entraîner le modèle avec n_estimators
    y_pred = rf.predict(X_test)  # Prédire sur l'ensemble de test
    score = accuracy_score(y_test, y_pred)  # Calculer la précision
    scores.append(score)

# Tracer l'évolution de la précision en fonction du nombre d'arbres
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, scores, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)
plt.title("Évolution de la précision en fonction du nombre d'arbres (n_estimators)", fontsize=14)
plt.xlabel("Nombre d'arbres (n_estimators)", fontsize=12)
plt.ylabel("Précision", fontsize=12)
plt.grid(True)
plt.show()
