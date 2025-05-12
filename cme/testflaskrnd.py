from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Charger le modèle et les encodeurs
objet = joblib.load('modele_rf_complet.pkl')
modele_rf = objet['model']
encoders = objet['encoders']

typologies = ['Fournitures', 'Services', 'Travaux']
garanties = ['Caution', 'Retenu', 'Aucune']
situations = ['Conforme', 'Non conforme']
blacklist = ['Oui', 'Non']

encoder_typologie = LabelEncoder().fit(typologies)
encoder_garantie = LabelEncoder().fit(garanties)
encoder_situation = LabelEncoder().fit(situations)
encoder_blacklist = LabelEncoder().fit(blacklist)

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
    return pd.DataFrame([dossier_encode])
CORS(app, resources={r"/predict": {"origins": "http://localhost:4200"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    donnees_traitees = traiter_dossier_texte(data)
    probabilites = modele_rf.predict_proba(donnees_traitees)
    resultat = {
        "prediction": "Visa Accordé" if probabilites[0][1] > probabilites[0][0] else "Visa Refusé",
        "confidence": {
            "Visa Accordé": f"{probabilites[0][1] * 100:.2f}%",
            "Visa Refusé": f"{probabilites[0][0] * 100:.2f}%"
        }
    }
    return jsonify(resultat)

if __name__ == '__main__':
    app.run(debug=True)
