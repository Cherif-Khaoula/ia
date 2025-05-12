# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Chargement des données
data = pd.read_csv('dossiers.csv')

# 2. Encodage des colonnes catégoriques
encoder = LabelEncoder()
colonnes_a_encoder = ['Typologie du marché', 'Garantie', 'Situation fiscale', 'Fournisseur blacklisté', 'Visa (Oui/Non)']
for col in colonnes_a_encoder:
    data[col] = encoder.fit_transform(data[col])

# 3. Séparation des données en X (features) et y (cible)
X = data.drop(columns=['Visa (Oui/Non)'])
y = data['Visa (Oui/Non)']

# 4. Division en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Définition des paramètres à tester
C_values = [0.1, 1, 10, 100]
gamma_values = ['scale', 0.01, 0.1, 1]

# 6. Entraînement des modèles avec différentes combinaisons
results = []

for C in C_values:
    for gamma in gamma_values:
        svm = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')  # class_weight équilibré
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy  # taux d'erreur = 1 - précision
        results.append({
            'C': C,
            'gamma': gamma,
            'accuracy': round(accuracy, 3),
            'error_rate': round(error_rate, 3)
        })

# 7. Visualisation des résultats
results_df = pd.DataFrame(results)

# Affichage du tableau des performances
print("Résultats des différentes combinaisons de paramètres :\n")
print(results_df)

# 8. Tracer le graphe
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='C', y='error_rate', hue='gamma', marker='o', palette='tab10')
plt.title("Impact des paramètres C et gamma sur le taux d'erreur (SVM)")
plt.xlabel("Paramètre C")
plt.ylabel("Taux d'erreur (1 - accuracy)")
plt.xscale('log')  # Echelle logarithmique pour C
plt.grid(True)
plt.legend(title='gamma')
plt.tight_layout()
plt.show()
