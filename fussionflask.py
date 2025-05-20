from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
from sklearn.neighbors import KNeighborsClassifier  # Importation de KNN

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

# Charger Random Forest
objet_rf = joblib.load('modele_rf_complet.pkl')
modele_rf = objet_rf['model']
encoders_rf = objet_rf['encoders']

# Charger le modèle XGBoost
objet_xgb = joblib.load('modele_xgboost.pkl')
modele_xgb = objet_xgb

# Charger SVM
modele_svm = joblib.load('modele_svm.pkl')

# Charger KNN
modele_knn = joblib.load('modele_knn.pkl')  # Charger le modèle KNN



# Charger les encodeurs et le scaler depuis un fichier séparé
encoders_et_scaler = joblib.load('encodeurs_scaler.pkl')
encoders = encoders_et_scaler['encoders']
scaler = encoders_et_scaler['scaler']

# Colonnes numériques à normaliser pour SVM et KNN
colonnes_numeriques = ['Montant du contrat', 'Délai de réalisation', 'Expérience fournisseur',
                       'Nombre de projets similaires', 'Notation interne', 'Chiffre d\'affaire']

# Fonction de traitement des données pour KNN
def traiter_knn(data):
    dossier_encode = {
        'Typologie du marché': encoders['Typologie du marché'].transform([data['Typologie du marché']])[0],
        'Montant du contrat': data['Montant du contrat'],
        'Garantie': encoders['Garantie'].transform([data['Garantie']])[0],
        'Délai de réalisation': data['Délai de réalisation'],
        'Expérience fournisseur': data['Expérience fournisseur'],
        'Nombre de projets similaires': data['Nombre de projets similaires'],
        'Notation interne': data['Notation interne'],
        'Chiffre d\'affaire': data['Chiffre d\'affaire'],
        'Situation fiscale': encoders['Situation fiscale'].transform([data['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders['Fournisseur blacklisté'].transform([data['Fournisseur blacklisté']])[0],
    }

    df = pd.DataFrame([dossier_encode])
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])
    return df

# Fonction de traitement des données pour XGBoost
def traiter_xgboost(data):
    dossier_encode = {
        'Typologie du marché': encoders['Typologie du marché'].transform([data['Typologie du marché']])[0],
        'Montant du contrat': data['Montant du contrat'],
        'Garantie': encoders['Garantie'].transform([data['Garantie']])[0],
        'Délai de réalisation': data['Délai de réalisation'],
        'Expérience fournisseur': data['Expérience fournisseur'],
        'Nombre de projets similaires': data['Nombre de projets similaires'],
        'Notation interne': data['Notation interne'],
        'Chiffre d\'affaire': data['Chiffre d\'affaire'],
        'Situation fiscale': encoders['Situation fiscale'].transform([data['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders['Fournisseur blacklisté'].transform([data['Fournisseur blacklisté']])[0],
    }

    df = pd.DataFrame([dossier_encode])
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])
    return df

# Fonction de traitement des données pour RF
def traiter_rf(data):
    return pd.DataFrame([{
        'Typologie du marché': encoders_rf['Typologie du marché'].transform([data['Typologie du marché']])[0],
        'Montant du contrat': data['Montant du contrat'],
        'Garantie': encoders_rf['Garantie'].transform([data['Garantie']])[0],
        'Délai de réalisation': data['Délai de réalisation'],
        'Expérience fournisseur': data['Expérience fournisseur'],
        'Nombre de projets similaires': data['Nombre de projets similaires'],
        'Notation interne': data['Notation interne'],
        'Chiffre d\'affaire': data['Chiffre d\'affaire'],
        'Situation fiscale': encoders_rf['Situation fiscale'].transform([data['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders_rf['Fournisseur blacklisté'].transform([data['Fournisseur blacklisté']])[0],
    }])
# Charger AdaBoost
modele_adaboost = joblib.load('modele_adaboost.pkl')

# Fonction de traitement des données pour SVM
def traiter_dossier_texte(dossier_texte):
    dossier_encode = {
        'Typologie du marché': encoders['Typologie du marché'].transform([dossier_texte['Typologie du marché']])[0],
        'Montant du contrat': dossier_texte['Montant du contrat'],
        'Garantie': encoders['Garantie'].transform([dossier_texte['Garantie']])[0],
        'Délai de réalisation': dossier_texte['Délai de réalisation'],
        'Expérience fournisseur': dossier_texte['Expérience fournisseur'],
        'Nombre de projets similaires': dossier_texte['Nombre de projets similaires'],
        'Notation interne': dossier_texte['Notation interne'],
        'Chiffre d\'affaire': dossier_texte['Chiffre d\'affaire'],
        'Situation fiscale': encoders['Situation fiscale'].transform([dossier_texte['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders['Fournisseur blacklisté'].transform([dossier_texte['Fournisseur blacklisté']])[0],
    }

    df = pd.DataFrame([dossier_encode])

    # Normalisation des colonnes numériques (comme à l'entraînement)
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])

    return df

# Fonction de traitement des données pour AdaBoost
def traiter_adaboost(data):
    dossier_encode = {
        'Typologie du marché': encoders['Typologie du marché'].transform([data['Typologie du marché']])[0],
        'Montant du contrat': data['Montant du contrat'],
        'Garantie': encoders['Garantie'].transform([data['Garantie']])[0],
        'Délai de réalisation': data['Délai de réalisation'],
        'Expérience fournisseur': data['Expérience fournisseur'],
        'Nombre de projets similaires': data['Nombre de projets similaires'],
        'Notation interne': data['Notation interne'],
        'Chiffre d\'affaire': data['Chiffre d\'affaire'],
        'Situation fiscale': encoders['Situation fiscale'].transform([data['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders['Fournisseur blacklisté'].transform([data['Fournisseur blacklisté']])[0],
    }

    df = pd.DataFrame([dossier_encode])
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])
    return df
modele_naive = joblib.load('modele_naive_bayes.pkl')
def traiter_naive(data):
    dossier_encode = {
        'Typologie du marché': encoders['Typologie du marché'].transform([data['Typologie du marché']])[0],
        'Montant du contrat': data['Montant du contrat'],
        'Garantie': encoders['Garantie'].transform([data['Garantie']])[0],
        'Délai de réalisation': data['Délai de réalisation'],
        'Expérience fournisseur': data['Expérience fournisseur'],
        'Nombre de projets similaires': data['Nombre de projets similaires'],
        'Notation interne': data['Notation interne'],
        'Chiffre d\'affaire': data['Chiffre d\'affaire'],
        'Situation fiscale': encoders['Situation fiscale'].transform([data['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoders['Fournisseur blacklisté'].transform([data['Fournisseur blacklisté']])[0],
    }

    df = pd.DataFrame([dossier_encode])
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])
    return df
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        # Random Forest
        rf_input = traiter_rf(data)
        proba_rf = modele_rf.predict_proba(rf_input)[0]
        rf_result = {
            "prediction": "Visa Accordé" if proba_rf[1] > proba_rf[0] else "Visa Refusé",
            "confidence": {
                "Visa Accordé": f"{proba_rf[1] * 100:.2f}%",
                "Visa Refusé": f"{proba_rf[0] * 100:.2f}%"
            }
        }

        # SVM
        donnees_traitees = traiter_dossier_texte(data)
        prediction_svm = modele_svm.predict(donnees_traitees)
        probabilite_svm = modele_svm.predict_proba(donnees_traitees)
        svm_result = {
            "prediction": "Visa Accordé" if prediction_svm[0] == 1 else "Visa Refusé",
            "confidence": {
                "Visa Accordé": f"{probabilite_svm[0][1] * 100:.2f}%",
                "Visa Refusé": f"{probabilite_svm[0][0] * 100:.2f}%"
            }
        }

        # XGBoost
        xgb_input = traiter_xgboost(data)
        proba_xgb = modele_xgb.predict_proba(xgb_input)[0]
        xgboost_result = {
            "prediction": "Visa Accordé" if proba_xgb[1] > proba_xgb[0] else "Visa Refusé",
            "confidence": {
                "Visa Accordé": f"{proba_xgb[1] * 100:.2f}%",
                "Visa Refusé": f"{proba_xgb[0] * 100:.2f}%"
            }
        }

                # KNN
        knn_input = traiter_knn(data)
        proba_knn = modele_knn.predict_proba(knn_input)[0]
        knn_result = {
            "prediction": "Visa Accordé" if proba_knn[1] > proba_knn[0] else "Visa Refusé",
            "confidence": {
                "Visa Accordé": f"{proba_knn[1] * 100:.2f}%",
                "Visa Refusé": f"{proba_knn[0] * 100:.2f}%"
            }
        }

        # AdaBoost
        adaboost_input = traiter_adaboost(data)
        proba_adaboost = modele_adaboost.predict_proba(adaboost_input)[0]
        adaboost_result = {
            "prediction": "Visa Accordé" if proba_adaboost[1] > proba_adaboost[0] else "Visa Refusé",
            "confidence": {
                "Visa Accordé": f"{proba_adaboost[1] * 100:.2f}%",
                "Visa Refusé": f"{proba_adaboost[0] * 100:.2f}%"
            }
        }

        # Naive Bayes
        naive_input = traiter_naive(data)
        proba_naive = modele_naive.predict_proba(naive_input)[0]
       

        naive_result = {
            "prediction": "Visa Accordé" if proba_naive[1] > proba_naive[0] else "Visa Refusé",

            "confidence": {
                "Visa Accordé": f"{proba_naive[1] * 100:.2f}%",
                "Visa Refusé": f"{proba_naive[0] * 100:.2f}%"
            }
        }

        return jsonify({
            "RandomForest": rf_result,
            "SVM": svm_result,
            "XGBoost": xgboost_result,
            "KNN": knn_result,
            "AdaBoost": adaboost_result,
            "NaiveBayes": naive_result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
