import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# === 7. Entraîner le modèle KNN ===
model_knn = KNeighborsClassifier(n_neighbors=5)  # Tu peux changer n_neighbors pour tester d'autres valeurs
model_knn.fit(X_train, y_train)

# === 8. Sauvegarder le modèle et les objets de prétraitement ===
joblib.dump(model_knn, "modele_knn.pkl")
joblib.dump({"encoders": encoders, "scaler": scaler}, "encodeurs_scaler.pkl")

print("✅ Modèle KNN entraîné et sauvegardé avec succès !")

# Évaluation du modèle
y_pred_knn = model_knn.predict(X_test)

# Affichage du rapport de classification avec des noms personnalisés
print("📋 Rapport de classification (KNN) :")
print(classification_report(y_test, y_pred_knn, target_names=["Non", "Oui"]))

# Affichage de la matrice de confusion
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=["Non", "Oui"])
disp_knn.plot(cmap=plt.cm.Blues)
plt.show()
