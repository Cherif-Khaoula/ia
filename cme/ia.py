import random
from faker import Faker
import pandas as pd

# Initialisation
fake = Faker()

typologies = ['Fournitures', 'Services', 'Travaux']

def generate_fake_data():
    typologie = random.choice(typologies)
    montant_contrat = random.randint(4000000, 10000000)
    garantie = random.choice(['Caution', 'Retenu', 'Aucune'])
    delai_realisation = random.choice([3, 4, 6, 8])
    experience_fournisseur = random.randint(1, 10)
    nb_projets_similaires = random.randint(0, 5)
    notation_interne = random.randint(10, 20)
    chiffre_affaire = random.randint(10000000, 50000000)
    situation_fiscale = random.choice(['Conforme', 'Non conforme'])
    fournisseur_blacklisté = random.choice(['Oui', 'Non'])

    return [typologie, montant_contrat, garantie, delai_realisation,
            experience_fournisseur, nb_projets_similaires, notation_interne,
            chiffre_affaire, situation_fiscale, fournisseur_blacklisté]

# Génération de 500 dossiers
dossiers_fictifs = [generate_fake_data() for _ in range(2000)]

# Colonnes finales
columns = ['Typologie du marché', 'Montant du contrat', 'Garantie', 'Délai de réalisation',
           'Expérience fournisseur', 'Nombre de projets similaires', 'Notation interne',
           'Chiffre d\'affaire', 'Situation fiscale', 'Fournisseur blacklisté']

# Création du DataFrame
df = pd.DataFrame(dossiers_fictifs, columns=columns)
def calculer_visa(row):
    # Règles bloquantes plus souples
    if row['Fournisseur blacklisté'] == 'Oui':
        return 'Non'
    if row['Situation fiscale'] == 'Non conforme' and row['Notation interne'] < 14:
        return 'Non'
    if row['Notation interne'] < 10:
        return 'Non'
    if row['Expérience fournisseur'] < 1 and row['Nombre de projets similaires'] == 0:
        return 'Non'
    
    # Règles spécifiques selon Typologie
    if row['Typologie du marché'] == 'Fournitures':
        if row['Montant du contrat'] > 8000000 and row['Garantie'] == 'Aucune':
            return 'Non'
        if row['Délai de réalisation'] > 8:
            return 'Non'

    elif row['Typologie du marché'] == 'Services':
        if row['Montant du contrat'] > 9000000 and row['Garantie'] == 'Aucune':
            return 'Non'
        if row['Délai de réalisation'] > 8:
            return 'Non'

    elif row['Typologie du marché'] == 'Travaux':
        if row['Montant du contrat'] > 9500000 and row['Garantie'] != 'Caution':
            return 'Non'
        if row['Délai de réalisation'] > 10:
            return 'Non'

    return 'Oui'

# Appliquer la fonction pour générer le visa
df['Visa (Oui/Non)'] = df.apply(calculer_visa, axis=1)
df['Visa (Oui/Non)'].value_counts()

# Sauvegarde
df.to_csv('dossiers.csv', index=False)

print(df.head())
