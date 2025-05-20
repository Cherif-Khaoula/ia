import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score , roc_curve, auc
import matplotlib.pyplot as plt

def train_svm():
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

    # Prédiction
    y_pred = svm.predict(X_test)

    # F1-score
    f1 = f1_score(y_test, y_pred, average='binary')

    # Rapport de classification
    print("=== Rapport de classification ===")
    print(classification_report(y_test, y_pred))

    # Sauvegarde du modèle
    joblib.dump(svm, 'modele_svm.pkl')
    print("Modèle SVM sauvegardé.")

    # Sauvegarde des encodeurs et du scaler
    joblib.dump({
        'encoders': encoders,
        'scaler': scaler
    }, 'encodeurs_scaler.pkl')
    print("Encodeurs et scaler sauvegardés.")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non", "Oui"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion - SVM")
    plt.show()
        # === 10. Tracé de la courbe ROC ===
    y_scores = svm.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"SVM (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Ligne de référence
    plt.xlabel("Taux de faux positifs (1 - Spécificité)")
    plt.ylabel("Taux de vrais positifs (Sensibilité)")
    plt.title("Courbe ROC - SVM")
    plt.legend(loc="lower right")
    plt.show()
    
    # ✅ Retour des résultats à la fin
    return {"nom": "SVM", "f1": f1}

if __name__ == "__main__":
    resultat = train_svm()
    print(resultat)
