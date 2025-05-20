from svm import train_svm  # ← adapte ce nom selon le nom réel du fichier
from rnd import train_rf
from knn import train_knn
from Naïve import train_nb
from XGBoost import train_xg
from AdaBoost import train_ad


import matplotlib.pyplot as plt

# Récupération des scores
results = [
    train_svm(),
    train_rf(),
    train_knn(),
    train_nb(),
    train_xg(),
    train_ad()
]

# Tracé du graphique comparatif
noms = [res['nom'] for res in results]
f1_scores = [res['f1'] for res in results]

plt.figure(figsize=(8,6))
bars = plt.bar(noms, f1_scores, color='lightcoral')
plt.title("Graphique des F1-scores comparés")
plt.xlabel("Algorithmes")
plt.ylabel("F1-score")

# Ajout des valeurs numériques au-dessus
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center')

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig("f1_scores_comparaison.png")
plt.show()
