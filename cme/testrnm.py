import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Charger le modèle Random Forest sauvegardé
modele_rf = joblib.load('modele_rf.pkl')

# Initialiser les encodeurs pour chaque colonne catégorique
typologies = ['Fournitures', 'Services', 'Travaux']
garanties = ['Caution', 'Retenu', 'Aucune']
situations = ['Conforme', 'Non conforme']
blacklist = ['Oui', 'Non']

encoder_typologie = LabelEncoder().fit(typologies)
encoder_garantie = LabelEncoder().fit(garanties)
encoder_situation = LabelEncoder().fit(situations)
encoder_blacklist = LabelEncoder().fit(blacklist)

# Fonction pour traiter l'entrée et effectuer les encodages
def traiter_dossier_texte(dossier_texte):
    dossier_encode = {
        'Typologie du marché': encoder_typologie.transform([dossier_texte['Typologie du marché']])[0],
        'Montant du contrat': dossier_texte['Montant du contrat'],
        'Garantie': encoder_garantie.transform([dossier_texte['Garantie']])[0],
        'Délai de réalisation': dossier_texte['Délai de réalisation'],
        'Expérience fournisseur': dossier_texte['Expérience fournisseur'],
        'Nombre de projets similaires': dossier_texte['Nombre de projets similaires'],
        'Notation interne': dossier_texte['Notation interne'],
        'Chiffre d\'affaire': dossier_texte['Chiffre d\'affaire'],
        'Situation fiscale': encoder_situation.transform([dossier_texte['Situation fiscale']])[0],
        'Fournisseur blacklisté': encoder_blacklist.transform([dossier_texte['Fournisseur blacklisté']])[0],
    }
    # Convertir en DataFrame pour la compatibilité avec le modèle
    return pd.DataFrame([dossier_encode])

# Exemple de dossier (avec des noms au lieu des codes)
nouveau_dossier = {
    'Typologie du marché': 'Fournitures',
    'Montant du contrat': 5000000,
    'Garantie': 'Aucune',
    'Délai de réalisation': 6,
    'Expérience fournisseur': 10,
    'Nombre de projets similaires': 2,
    'Notation interne': 13,
    'Chiffre d\'affaire': 45000000,
    'Situation fiscale': 'Conforme',
    'Fournisseur blacklisté': 'Non'
}

# Traiter le dossier pour l'encodage
donnee_traitee = traiter_dossier_texte(nouveau_dossier)

# Prédire avec le modèle chargé (retourne les probabilités)
probabilites = modele_rf.predict_proba(donnee_traitee)

# Affichage du résultat avec pourcentage
proba_visa_accepte = probabilites[0][1] * 100  # Probabilité de la classe "Visa Accordé"
proba_visa_refuse = probabilites[0][0] * 100  # Probabilité de la classe "Visa Refusé"

print(f"Prédiction pour le dossier : {'Visa Accordé' if proba_visa_accepte > proba_visa_refuse else 'Visa Refusé'}")
print(f"Confiance du modèle : Visa Accordé -> {proba_visa_accepte:.2f}%, Visa Refusé -> {proba_visa_refuse:.2f}%")
