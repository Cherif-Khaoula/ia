from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

# Charger le modèle XGBoost
modele_xgb = joblib.load('modele_xgboost.pkl')

# Charger les encodeurs et le scaler
pretraitements = joblib.load('encodeurs_scaler.pkl')
encoders = pretraitements['encoders']
scaler = pretraitements['scaler']

# Fonction pour traiter les données reçues
def traiter_dossier_texte(dossier_texte):
    # Encodage manuel
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

    colonnes_numeriques = ['Montant du contrat', 'Délai de réalisation', 'Expérience fournisseur',
                           'Nombre de projets similaires', 'Notation interne', 'Chiffre d\'affaire']
    df[colonnes_numeriques] = scaler.transform(df[colonnes_numeriques])

    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Données manquantes"}), 400

        donnees_traitees = traiter_dossier_texte(data)

        if isinstance(donnees_traitees, pd.DataFrame):
            prediction = modele_xgb.predict(donnees_traitees)
            proba = modele_xgb.predict_proba(donnees_traitees)

            resultat = {
                "prediction": "Visa Accordé" if prediction[0] == 1 else "Visa Refusé",
                "confidence": {
                    "Visa Accordé": f"{proba[0][1] * 100:.2f}%" if prediction[0] == 1 else "0%",
                    "Visa Refusé": f"{proba[0][0] * 100:.2f}%" if prediction[0] == 0 else "0%"
                }
            }
            return jsonify(resultat)
        else:
            return jsonify({"error": "Erreur de format"}), 400

    except Exception as e:
        return jsonify({"error": f"Erreur dans la prédiction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
