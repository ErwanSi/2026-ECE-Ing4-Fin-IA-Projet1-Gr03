import random
import numpy as np

def creation_demineur(taille, bombes):
    demineur = np.zeros((taille, taille), dtype=int)

    i = 0
    while i < bombes:
        x = random.randint(0, taille-1)
        y = random.randint(0, taille-1)

        if demineur[x][y] == 0:
            #Le niméro 9 représente une mine
            demineur[x][y] = 9
            i+=1


    return demineur

def ajout_nbr_casevide(demineur, i, j):
    if demineur[i][j] != 9:
        demineur[i][j]+=1

def mine_adjacent(demineur, taille):

    for i in range(taille):
        for j in range(taille):
            if demineur[i][j] == 9:
                if i != 0:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i-1, j-1)
                    ajout_nbr_casevide(demineur, i-1, j)
                    if j != taille-1:
                        ajout_nbr_casevide(demineur, i-1, j+1)
                if j != 0:
                    ajout_nbr_casevide(demineur, i, j-1)
                if j != taille-1:
                    ajout_nbr_casevide(demineur, i, j+1)
                if i != taille-1:
                    if j != 0:
                        ajout_nbr_casevide(demineur, i+1, j-1)
                    ajout_nbr_casevide(demineur, i+1, j)
                    if j != taille-1:
                        ajout_nbr_casevide(demineur, i+1, j+1)
    return demineur


def affichage(demineur, taille, mask):
    for i in range(taille):
        for k in range(taille*4):
            print("_", end='')
            if k == taille*4 - 1:
                print("_", end='\n')

        for j in range(taille):
            if mask[i][j]:
                print("#", end=' | ')
            else:
                print(demineur[i][j], end=' | ')
            if j == taille-1:
                print("", end='\n')

    print()
    print()

def jeu(demineur, mask, taille, x, y):
    if demineur[x][y] == 9:
        print("Perdu")

    if demineur[x][y] == 0 and mask[x][y] == 1:
        mask[x][y] = 0
        if x != 0:
            if y != 0:
                jeu(demineur, mask, taille, x-1, y-1)
            jeu(demineur, mask, taille, x-1, y)
            if y != taille-1:
                jeu(demineur, mask, taille, x-1, y+1)
        if y != 0:
            jeu(demineur, mask, taille, x, y-1)
        if y != taille-1:
            jeu(demineur, mask, taille, x, y+1)
        if x != taille-1:
            if y != 0:
                jeu(demineur, mask, taille, x+1, y-1)
            jeu(demineur, mask, taille, x+1, y)
            if y != taille-1:
                jeu(demineur, mask, taille, x+1, y+1)

    mask[x][y] = 0

taille = 10
bombes = 7

demineur = creation_demineur(taille, bombes)
print(demineur)
demineur = mine_adjacent(demineur, taille)
print(demineur)

mask = np.ones(demineur.shape, dtype=int)

affichage(demineur, taille, mask)

jeu(demineur, mask, taille, 3, 3)

affichage(demineur, taille, mask)
