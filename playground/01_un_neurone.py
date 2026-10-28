#!/usr/bin/env python3
# Étape 1 : un seul neurone apprend la règle "ET"
# (la réponse vaut 1 seulement si A et B valent 1 tous les deux)

import math
import random

# --- Les exemples corrigés : (A, B, bonne réponse) ---
exemples = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

# --- Les réglages du neurone, choisis au hasard au départ ---
random.seed(42)                  # pour avoir le même hasard à chaque lancement
poids_a = random.uniform(-1, 1)  # importance de l'entrée A
poids_b = random.uniform(-1, 1)  # importance de l'entrée B
biais = random.uniform(-1, 1)    # un "coup de pouce" de départ

vitesse = 0.5 # de combien on tourne les boutons à chaque correction


def ecraser(x):
    """Transforme n'importe quel nombre en un nombre entre 0 et 1.
    Très négatif -> proche de 0, très positif -> proche de 1.
    (Son vrai nom : la fonction sigmoïde.)"""
    return 1 / (1 + math.exp(-x))


def neurone(a, b):
    """Le neurone : on multiplie, on additionne, on écrase."""
    total = a * poids_a + b * poids_b + biais
    return ecraser(total)

print("Réglages initiaux :")
print(f"  poids_a = {poids_a:.2f}, poids_b = {poids_b:.2f}, biais = {biais:.2f}")
print(f"  vitesse = {vitesse}")
print()
# --- L'apprentissage : on répète la boucle plein de fois ---
for tour in range(1, 2001):
    erreur_totale = 0

    for a, b, bonne_reponse in exemples:
        # 1. Le neurone donne sa réponse
        reponse = neurone(a, b)

        # 2. On mesure l'écart avec la bonne réponse
        ecart = reponse - bonne_reponse
        erreur_totale += abs(ecart)

        # 3. On corrige chaque bouton un tout petit peu.
        #    Si l'entrée valait 0, elle n'a pas participé : pas de correction.
        poids_a -= vitesse * ecart * a
        poids_b -= vitesse * ecart * b
        biais -= vitesse * ecart

    if tour in (1, 10, 100, 500, 1000, 2000):
        print(f"Tour {tour:4d} : erreur moyenne = {erreur_totale / 4:.3f}")

# --- Résultat final ---
print()
print("Ce que le neurone a appris :")
for a, b, bonne_reponse in exemples:
    reponse = neurone(a, b)
    print(f"  {a} || {b} -> {reponse:.2f}  (arrondi : {round(reponse)}, attendu : {bonne_reponse})")
print()
print(f"Réglages trouvés : poids_a = {poids_a:.2f}, poids_b = {poids_b:.2f}, biais = {biais:.2f}")
