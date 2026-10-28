#!/usr/bin/env python3
# Étape 2b : le réseau APPREND tout seul la règle "différents"
# On part de poids au hasard, et on utilise la rétropropagation :
# "chacun reçoit sa part de responsabilité dans l'erreur et se corrige".

import math
import random

exemples = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

random.seed(42)  # essaie aussi 1 ou 3 : que se passe-t-il ?
NB_CACHES = 10 # nombre de neurones dans la couche cachée : essaie 3, 4...
vitesse = 0.5
NB_TOURS = 10000


def hasard():
    return random.uniform(-1, 1)


# --- Couche cachée : NB_CACHES neurones, chacun a 2 poids (pour A et B) et 1 biais ---
caches_poids = [[hasard(), hasard()] for _ in range(NB_CACHES)]
caches_biais = [hasard() for _ in range(NB_CACHES)]

# --- Neurone de sortie : 1 poids par neurone caché, et 1 biais ---
sortie_poids = [hasard() for _ in range(NB_CACHES)]
sortie_biais = hasard()


def ecraser(x):
    """La fonction sigmoïde : ramène n'importe quel nombre entre 0 et 1."""
    return 1 / (1 + math.exp(-x))


def neurone(entrees, poids, biais):
    """Multiplier, additionner, ajouter le biais, écraser."""
    total = biais
    for entree, p in zip(entrees, poids):
        total += entree * p
    return ecraser(total)


def reseau(a, b):
    """Fait passer A et B dans tout le réseau.
    Renvoie les indices des neurones cachés ET la réponse finale :
    on a besoin des deux pour savoir qui corriger."""
    indices = [neurone([a, b], caches_poids[i], caches_biais[i]) for i in range(NB_CACHES)]
    reponse = neurone(indices, sortie_poids, sortie_biais)
    return indices, reponse

print()
print(f"Nombre de neurones cachés : {NB_CACHES}")
print(f"Vitesse d'apprentissage : {vitesse}")
print()

for tour in range(1, NB_TOURS + 1):
    erreur_totale = 0

    for a, b, bonne_reponse in exemples:
        # 1. Le réseau donne sa réponse
        indices, reponse = reseau(a, b)

        # 2. Faute du neurone de sortie : pareil qu'à l'étape 1
        faute_sortie = reponse - bonne_reponse
        erreur_totale += abs(faute_sortie)

        # 3. La sortie "remonte la faute" vers chaque neurone caché.
        #    Part de faute = faute de la sortie
        #                  x importance que la sortie lui accordait (son poids)
        #                  x à quel point ce neurone hésitait  [indice x (1 - indice)]
        #    Un neurone qui hésite (indice proche de 0.5) est facile à faire changer d'avis ;
        #    un neurone déjà très sûr (proche de 0 ou 1) bouge peu.
        fautes_caches = []
        for i in range(NB_CACHES):
            hesitation = indices[i] * (1 - indices[i])
            fautes_caches.append(faute_sortie * sortie_poids[i] * hesitation)

        # 4. Tout le monde se corrige (même règle qu'à l'étape 1 :
        #    bouton -= vitesse x faute x ce qui est entré dans le neurone)
        for i in range(NB_CACHES):
            sortie_poids[i] -= vitesse * faute_sortie * indices[i]
        sortie_biais -= vitesse * faute_sortie

        for i in range(NB_CACHES):
            caches_poids[i][0] -= vitesse * fautes_caches[i] * a
            caches_poids[i][1] -= vitesse * fautes_caches[i] * b
            caches_biais[i] -= vitesse * fautes_caches[i]

    if tour in (1, 100, 1000, 2000, 5000, 10000):
        print(f"Tour {tour:5d} : erreur moyenne = {erreur_totale / 4:.3f}")

# --- Résultat final ---
print()
entete = " A  B |" + "".join(f" caché {i + 1} |" for i in range(NB_CACHES)) + " réponse | attendu"
print(entete)
print("-" * len(entete))
for a, b, attendu in exemples:
    indices, reponse = reseau(a, b)
    if 0.3 < reponse < 0.7:
        verdict = "🤷 hésite"  # l'arrondi peut tomber juste par chance : on ne compte pas ça comme réussi
    elif round(reponse) == attendu:
        verdict = "✅"
    else:
        verdict = "❌"
    colonnes = "".join(f"   {x:.2f}   |" for x in indices)
    print(f" {a}  {b} |{colonnes}  {reponse:.2f}   |   {attendu}  {verdict}")

print()
print("Réglages trouvés :")
for i in range(NB_CACHES):
    print(f"  neurone caché {i + 1} : poids = {caches_poids[i][0]:.2f}, {caches_poids[i][1]:.2f}  biais = {caches_biais[i]:.2f}")
poids_sortie_texte = ", ".join(f"{p:.2f}" for p in sortie_poids)
print(f"  neurone de sortie : poids = {poids_sortie_texte}  biais = {sortie_biais:.2f}")
