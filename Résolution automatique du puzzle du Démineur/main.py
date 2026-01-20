import Demineur as d
import numpy as np


def choix_chiffre(max_val, coo):
    while True:
        try:
            n = int(input(f"Choisir une coordonné pour {coo} entre 0 et {max_val} : "))
            if 0 <= n <= max_val:
                break
            else:
                print("❌ Le nombre n’est pas dans la plage.")
        except ValueError:
            print("❌ Entre un nombre valide.")
    return n


taille = 10
bombes = 7

y = choix_chiffre(taille, "x")
x = choix_chiffre(taille, "y")

demineur = d.creation_demineur(taille, bombes, x, y)
print(demineur)
demineur = d.mine_adjacent(demineur, taille)
print(demineur)
mask = np.ones(demineur.shape, dtype=int)
d.jeu(demineur, mask, taille, x, y)
d.affichage(demineur, taille, mask)

while d.fin_jeu(demineur, mask, taille, bombes):
    y = choix_chiffre(taille, "x")
    x = choix_chiffre(taille, "y")

    d.jeu(demineur, mask, taille, x, y)
    d.affichage(demineur, taille, mask)