#!/usr/bin/env python3
# Étape 2a : un petit réseau de 3 neurones résout la règle "différents"
# Ici, PAS d'apprentissage : on règle les poids à la main
# avec ceux que nos neurones ont trouvés à l'étape 1.

import math


def ecraser(x):
    """La fonction sigmoïde : ramène n'importe quel nombre entre 0 et 1."""
    return 1 / (1 + math.exp(-x))


def neurone(entrees, poids, biais):
    """Un neurone général : autant d'entrées qu'on veut.
    On multiplie chaque entrée par son poids, on additionne, on ajoute le biais, on écrase."""
    total = biais
    for entree, p in zip(entrees, poids):
        total += entree * p
    return ecraser(total)


def reseau(a, b):
    # --- Couche cachée : deux neurones qui repèrent des indices ---
    # Réglages trouvés par le neurone "OU" à l'étape 1
    indice_ou = neurone([a, b], poids=[11.49, 11.49], biais=-5.29)
    # Réglages trouvés par le neurone "ET" à l'étape 1
    indice_et = neurone([a, b], poids=[10.26, 10.26], biais=-15.55)

    # --- Couche de sortie : un neurone qui combine les indices ---
    # "OU" pousse vers 1 (poids positif), "ET" pousse vers 0 (poids négatif)
    reponse = neurone([indice_ou, indice_et], poids=[10, -10], biais=-5)

    return indice_ou, indice_et, reponse


exemples = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

print(" A  B | indice OU | indice ET | réponse | attendu")
print("------+-----------+-----------+---------+--------")
for a, b, attendu in exemples:
    ou, et, reponse = reseau(a, b)
    ok = "✅" if round(reponse) == attendu else "❌"
    print(f" {a}  {b} |   {ou:.2f}    |   {et:.2f}    |  {reponse:.2f}   |   {attendu}  {ok}")
