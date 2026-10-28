#!/usr/bin/env python3
# Étape 3 : EXACTEMENT le même réseau que 03_reseau_qui_apprend.py,
# mais les poids sont rangés dans des tableaux NumPy.
# Les boucles "for i in range(NB_CACHES)" disparaissent :
# NumPy fait le calcul pour tous les neurones d'un coup.

import random

import numpy as np

exemples = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

random.seed(42)
NB_CACHES = 3
vitesse = 0.5
NB_TOURS = 10000


def hasard():
    return random.uniform(-1, 1)


# On tire le hasard dans le même ordre que dans 03 : même départ, donc même résultat.

# Couche cachée : un tableau de NB_CACHES lignes et 2 colonnes.
#   ligne i = les 2 poids du neurone caché i (colonne 0 pour A, colonne 1 pour B)
poids_caches = np.array([[hasard(), hasard()] for _ in range(NB_CACHES)])
biais_caches = np.array([hasard() for _ in range(NB_CACHES)])

# Neurone de sortie : une liste de NB_CACHES poids (un par neurone caché)
poids_sortie = np.array([hasard() for _ in range(NB_CACHES)])
biais_sortie = hasard()


def ecraser(x):
    """La sigmoïde. Avec NumPy, x peut être un tableau entier :
    elle s'applique alors à chaque case d'un coup."""
    return 1 / (1 + np.exp(-x))


def reseau(entree):
    """entree est un tableau [A, B]."""
    # "@" = pour chaque ligne du tableau : multiplier case par case avec l'entrée, puis additionner.
    # C'est exactement "total = A * poids_a + B * poids_b", fait pour tous les neurones à la fois.
    indices = ecraser(poids_caches @ entree + biais_caches)
    reponse = ecraser(poids_sortie @ indices + biais_sortie)
    return indices, reponse


print(f"Nombre de neurones cachés : {NB_CACHES}")
print(f"Vitesse d'apprentissage : {vitesse}")
print()

for tour in range(1, NB_TOURS + 1):
    erreur_totale = 0

    for a, b, bonne_reponse in exemples:
        entree = np.array([a, b])

        # 1. Le réseau donne sa réponse
        indices, reponse = reseau(entree)

        # 2. Faute du neurone de sortie
        faute_sortie = reponse - bonne_reponse
        erreur_totale += abs(faute_sortie)

        # 3. Part de faute de chaque neurone caché (tous d'un coup) :
        #    faute de la sortie x son poids x son hésitation
        fautes_caches = faute_sortie * poids_sortie * indices * (1 - indices)

        # 4. Tout le monde se corrige
        poids_sortie -= vitesse * faute_sortie * indices
        biais_sortie -= vitesse * faute_sortie
        # np.outer fabrique le tableau "faute du neurone i x entrée j" pour toutes les cases :
        # c'est la même règle "bouton -= vitesse x faute x ce qui est entré", pour tout le tableau.
        poids_caches -= vitesse * np.outer(fautes_caches, entree)
        biais_caches -= vitesse * fautes_caches

    if tour in (1, 100, 1000, 2000, 5000, 10000):
        print(f"Tour {tour:5d} : erreur moyenne = {erreur_totale / 4:.3f}")

# --- Résultat final ---
print()
entete = " A  B |" + "".join(f" caché {i + 1} |" for i in range(NB_CACHES)) + " réponse | attendu"
print(entete)
print("-" * len(entete))
for a, b, attendu in exemples:
    indices, reponse = reseau(np.array([a, b]))
    if 0.3 < reponse < 0.7:
        verdict = "🤷 hésite"
    elif round(reponse) == attendu:
        verdict = "✅"
    else:
        verdict = "❌"
    colonnes = "".join(f"   {x:.2f}   |" for x in indices)
    print(f" {a}  {b} |{colonnes}  {reponse:.2f}   |   {attendu}  {verdict}")

print()
print("Réglages trouvés :")
print("  neurones cachés (une ligne par neurone : poids A, poids B) :")
print(np.round(poids_caches, 2))
print("  biais des neurones cachés :", np.round(biais_caches, 2))
print("  neurone de sortie : poids =", np.round(poids_sortie, 2), f" biais = {biais_sortie:.2f}")
