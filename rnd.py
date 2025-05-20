# Importation des bibliothèques nécessaires
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score , roc_curve, auc
import matplotlib.pyplot as plt
def train_rf():
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
 # F1-score
 f1 = f1_score(y_test, y_pred, average='binary')
 # Sauvegarde du modèle Random Forest et des encodeurs dans un fichier
 joblib.dump({'model': rf, 'encoders': encoders}, 'modele_rf_complet.pkl')
 print("Modèle Random Forest et encodeurs sauvegardés sous le nom 'modele_rf_complet.pkl'.")

 # Chargement du modèle sauvegardé pour utilisation future (par exemple, dans un autre script)
 modele_charge = joblib.load('modele_rf_complet.pkl')
 print("Modèle chargé avec succès.")  

 # Affichage de la matrice de confusion
 cm_knn = confusion_matrix(y_test, y_pred)
 disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Non", "Oui"])
 disp_knn.plot(cmap=plt.cm.Blues)
 plt.title("Matrice de confusion -Random Forest")
 plt.show()
    # === 10. Tracé de la courbe ROC ===
 y_scores = rf.predict_proba(X_test)[:, 1]
 fpr, tpr, _ = roc_curve(y_test, y_scores)
 roc_auc = auc(fpr, tpr)

 plt.figure(figsize=(8, 6))
 plt.plot(fpr, tpr, color='blue', label=f"Random Forest (AUC = {roc_auc:.2f})")
 plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Ligne de référence
 plt.xlabel("Taux de faux positifs (1 - Spécificité)")
 plt.ylabel("Taux de vrais positifs (Sensibilité)")
 plt.title("Courbe ROC - Random Forest")
 plt.legend(loc="lower right")
 plt.show()
 # ✅ Retour des résultats à la fin
 return {"nom": "RandomForest", "f1": f1}

if __name__ == "__main__":
    resultat = train_rf()
    print(resultat)

