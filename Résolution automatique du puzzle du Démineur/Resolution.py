import numpy as np
import Demineur as d

# Cette fonction permet d'avoir un plateau de jeu comme le joueur voit
def plateau(demineur, mask, taille):
    plat = np.zeros(demineur.shape, dtype=int)
    for x in range(taille):
        for y in range(taille):
            if mask[x][y] == 1:
                plat[x][y] = -1
            else:
                plat[x][y] = demineur[x][y]
    return plat

def case_decouverte(plat, x, y):
    if plat[x][y] == -1:
        return 0
    return 1

def bordure(plat, taille):
    for i in range(taille):
        for j in range(taille):
            if plat[i][j] >= 0:
                # Ligne du dessus
                if i != 0:
                    if j != 0:
                        if not case_decouverte(plat, i-1, j-1):
                            plat[i-1][j-1] = -2
                    if not case_decouverte(plat, i - 1, j):
                        plat[i-1][j] = -2
                    if j != taille - 1:
                        if not case_decouverte(plat, i - 1, j + 1):
                            plat[i-1][j+1] = -2
                # Même ligne
                if j != 0:
                    if not case_decouverte(plat, i, j - 1):
                        plat[i][j-1] = -2
                if j != taille - 1:
                    if not case_decouverte(plat, i, j + 1):
                        plat[i][j+1] = -2
                # Ligne du dessous
                if i != taille - 1:
                    if j != 0:
                        if not case_decouverte(plat, i + 1, j - 1):
                            plat[i+1][j-1] = -2
                    if not case_decouverte(plat, i + 1, j):
                        plat[i+1][j] = -2
                    if j != taille - 1:
                        if not case_decouverte(plat, i + 1, j + 1):
                            plat[i+1][j+1] = -2

    print(plat)
    return plat

def liste_bordure(plat):
    variables = []
    for i in range(len(plat)):
        for j in range(len(plat)):
            if plat[i][j] == -2:
                variables.append((i, j))
    return variables

demineur = d.creation_demineur(10, 20, 3,3)
mask = np.ones(demineur.shape, dtype=int)

demineur = d.mine_adjacent(demineur, 10)

d.jeu(demineur, mask, 10, 3, 3)

d.affichage(demineur,10, mask)

plat = plateau(demineur, mask, 10)
print(plat)
bordure(plat, 10)
print(liste_bordure(plat))