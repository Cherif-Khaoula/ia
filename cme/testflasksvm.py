from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

# Charger le modèle SVM
modele_svm = joblib.load('modele_svm.pkl')

# Charger les encodeurs et le scaler depuis un fichier séparé
encoders_et_scaler = joblib.load('encodeurs_scaler.pkl')
encoders = encoders_et_scaler['encoders']
scaler = encoders_et_scaler['scaler']

# Fonction de traitement des données
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
    colonnes_numeriques = ['Montant du contrat', 'Délai de réalisation', 'Expérience fournisseur',
                           'Nombre de projets similaires', 'Notation interne', 'Chiffre d\'affaire']
    
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])
    
    return df


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Récupère les données JSON envoyées par Postman
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        # Traiter les données avant de les passer au modèle
        donnees_traitees = traiter_dossier_texte(data)

        # Vérifie que les données sont bien sous forme de DataFrame avant d'appliquer le modèle
        if isinstance(donnees_traitees, pd.DataFrame):
            prediction = modele_svm.predict(donnees_traitees)  # Utilisation du modèle SVM pour prédire
            probabilite = modele_svm.predict_proba(donnees_traitees)  # Si tu veux la probabilité

            # Renvoi de la prédiction et de la probabilité associée
            resultat = {
                "prediction": "Visa Accordé" if prediction[0] == 1 else "Visa Refusé",  # 1 = Visa Accordé, 0 = Visa Refusé
                "confidence": {
                    "Visa Accordé": f"{probabilite[0][1] * 100:.2f}%" if prediction[0] == 1 else "0%",
                    "Visa Refusé": f"{probabilite[0][0] * 100:.2f}%" if prediction[0] == 0 else "0%"
                }
            }
            return jsonify(resultat)
        else:
            return jsonify({"error": "Données invalides après traitement"}), 400

    except Exception as e:
        return jsonify({"error": f"Erreur dans la prédiction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
