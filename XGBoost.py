import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
def train_xg():
 # === 1. Charger les données ===
 df = pd.read_csv("dossiers.csv")  # Remplace par ton fichier CSV

 # === 2. Convertir la cible 'Visa (Oui/Non)' en binaire ===
 df['Visa'] = df["Visa (Oui/Non)"].map({'Oui': 1, 'Non': 0})
 df = df.drop(columns=["Visa (Oui/Non)"])

 # === 3. Séparer les features et la cible ===
 X = df.drop("Visa", axis=1)
 y = df["Visa"]

 # === 4. Encoder les colonnes catégorielles ===
 colonnes_cat = ['Typologie du marché', 'Garantie', 'Situation fiscale', 'Fournisseur blacklisté']
 encoders = {}

 for col in colonnes_cat:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

 # === 5. Normaliser les colonnes numériques ===
 colonnes_num = ['Montant du contrat', 'Délai de réalisation', 'Expérience fournisseur',
                'Nombre de projets similaires', 'Notation interne', "Chiffre d'affaire"]

 scaler = StandardScaler()
 X[colonnes_num] = scaler.fit_transform(X[colonnes_num])

 # === 6. Split en train/test ===
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # === 7. Entraîner le modèle XGBoost ===
 model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
 model.fit(X_train, y_train)

 # === 8. Sauvegarder le modèle et les objets de prétraitement ===
 joblib.dump(model, "modele_xgboost.pkl")
 joblib.dump({"encoders": encoders, "scaler": scaler}, "encodeurs_scaler.pkl")

 print("✅ Modèle XGBoost entraîné et sauvegardé avec succès !")


 # Évaluation du modèle
 y_pred = model.predict(X_test)
 y_scores = model.predict_proba(X_test)[:, 1]
 # F1-score
 f1 = f1_score(y_test, y_pred, average='binary')
 # Affichage du rapport de classification avec des noms personnalisés
 print("📋 Rapport de classification :")
 print(classification_report(y_test, y_pred, target_names=["Non", "Oui"]))

 # Affichage de la matrice de confusion
 cm_adaboost = confusion_matrix(y_test, y_pred)
 disp_adaboost = ConfusionMatrixDisplay(confusion_matrix=cm_adaboost, display_labels=["Non", "Oui"])
 disp_adaboost.plot(cmap=plt.cm.Blues)
 plt.title("Matrice de confusion -XGBoost")
 plt.show()
 # === 10. Tracé de la courbe ROC ===
 fpr, tpr, _ = roc_curve(y_test, y_scores)
 roc_auc = auc(fpr, tpr)

 plt.figure(figsize=(8, 6))
 plt.plot(fpr, tpr, color='blue', label=f"XGBoost (AUC = {roc_auc:.2f})")
 plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Ligne de référence
 plt.xlabel("Taux de faux positifs (1 - Spécificité)")
 plt.ylabel("Taux de vrais positifs (Sensibilité)")
 plt.title("Courbe ROC - XGBoost")
 plt.legend(loc="lower right")
 plt.show()
  # ✅ Retour des résultats à la fin
 return {"nom": "XGBoost ", "f1": f1}

if __name__ == "__main__":
    resultat = train_xg()
    print(resultat)
